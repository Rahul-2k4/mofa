//! Skills Manager - SDK 层统一 API

use super::{
    DisclosureController, HubCacheStatus, HubSkillCatalogEntry, ManagedHubSkillRecord,
    RequirementCheck, SkillHubClient, SkillHubClientConfig, SkillMetadata, SkillParser,
    SkillRequirements,
};
use crate::llm::context::SkillsManager as PromptSkillsManager;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Skills Manager - SDK 层统一 API
///
/// 提供 Skills 的管理和查询接口，支持渐进式披露、多目录搜索和依赖检查。
#[derive(Debug, Clone)]
pub struct SkillsManager {
    controller: DisclosureController,
    hub: Option<SkillHubClient>,
}

impl SkillsManager {
    /// 创建新的 Skills Manager（单目录）
    ///
    /// # Arguments
    ///
    /// * `skills_dir` - Skills 目录路径
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use mofa_sdk::skills::SkillsManager;
    ///
    /// let manager = SkillsManager::new("./skills").unwrap();
    /// ```
    pub fn new(skills_dir: impl AsRef<Path>) -> anyhow::Result<Self> {
        let skills_dir = skills_dir.as_ref();

        // 不要求目录必须存在（支持空目录）
        let mut controller = DisclosureController::new(skills_dir);
        if skills_dir.exists() {
            controller.scan_metadata()?;
        }

        Ok(Self {
            controller,
            hub: None,
        })
    }

    /// 创建支持多目录搜索的 Skills Manager
    ///
    /// # Arguments
    ///
    /// * `search_dirs` - Skills 目录列表，按优先级排序
    pub fn with_search_dirs(search_dirs: Vec<PathBuf>) -> anyhow::Result<Self> {
        let controller = DisclosureController::with_search_dirs(search_dirs);
        let mut manager = Self {
            controller,
            hub: None,
        };
        manager.rescan()?;
        Ok(manager)
    }

    /// 创建支持 Skills Hub 的 Skills Manager
    pub fn with_hub(
        skills_dir: impl AsRef<Path>,
        hub_config: SkillHubClientConfig,
    ) -> anyhow::Result<Self> {
        Self::with_search_dirs_and_hub(vec![skills_dir.as_ref().to_path_buf()], hub_config)
    }

    /// 创建支持 Skills Hub 的多目录 Skills Manager
    pub fn with_search_dirs_and_hub(
        mut search_dirs: Vec<PathBuf>,
        hub_config: SkillHubClientConfig,
    ) -> anyhow::Result<Self> {
        let hub = SkillHubClient::new(hub_config)?;
        if !search_dirs.contains(&hub.config().managed_root) {
            search_dirs.push(hub.config().managed_root.clone());
        }

        let controller = DisclosureController::with_search_dirs(search_dirs);
        let mut manager = Self {
            controller,
            hub: Some(hub),
        };
        manager.rescan()?;
        Ok(manager)
    }

    /// 使用环境变量中的默认 Hub 配置创建 Skills Manager
    pub fn with_env_hub(skills_dir: impl AsRef<Path>) -> anyhow::Result<Self> {
        Self::with_hub(skills_dir, SkillHubClientConfig::from_env())
    }

    /// 查找内置 skills 目录
    pub fn find_builtin_skills() -> Option<PathBuf> {
        DisclosureController::find_builtin_skills()
    }

    /// 获取已配置的 Hub 客户端
    pub fn hub_client(&self) -> Option<&SkillHubClient> {
        self.hub.as_ref()
    }

    /// 获取系统提示（第1层：仅元数据）
    pub fn build_system_prompt(&self) -> String {
        self.controller.build_system_prompt()
    }

    /// 获取系统提示（异步版本）
    pub async fn build_system_prompt_async(&self) -> String {
        self.build_system_prompt()
    }

    /// 加载 Skill 的 SKILL.md 内容（第2层）
    pub fn load_skill(&self, name: &str) -> Option<String> {
        if let Some(skill_md) = self.local_skill_markdown_path(name) {
            return read_skill_markdown(&skill_md);
        }

        if let Some(skill_md) = self.managed_skill_markdown_path(name) {
            if let Some(hub) = &self.hub {
                let _ = hub.touch_last_used(name);
            }
            return read_skill_markdown(&skill_md);
        }

        None
    }

    /// 加载 Skill 的 SKILL.md 内容（异步版本）
    pub async fn load_skill_async(&self, name: &str) -> Option<String> {
        if let Some(skill_md) = self.local_skill_markdown_path(name) {
            return read_skill_markdown_async(&skill_md).await;
        }

        if let Some(skill_md) = self.managed_skill_markdown_path(name) {
            if let Some(hub) = &self.hub {
                let _ = hub.touch_last_used(name);
            }
            return read_skill_markdown_async(&skill_md).await;
        }

        if let Some(hub) = &self.hub {
            if hub.ensure_installed(name).await.ok().flatten().is_some() {
                if let Some(skill_md) = self.managed_skill_markdown_path(name) {
                    return read_skill_markdown_async(&skill_md).await;
                }
            }
        }

        None
    }

    /// 加载多个 Skills 的内容用于上下文
    pub async fn load_skills_for_context(&self, skill_names: &[String]) -> String {
        let mut parts = Vec::new();

        for name in skill_names {
            if let Some(content) = self.load_skill_async(name).await
                && !content.is_empty()
            {
                parts.push(format!("### Skill: {}\n\n{}", name, content));
            }
        }

        parts.join("\n\n---\n\n")
    }

    /// 获取标记为 always 的技能名称列表
    pub fn get_always_skills(&self) -> Vec<String> {
        self.controller.get_always_skills()
    }

    /// 获取标记为 always 的技能名称列表（异步版本）
    pub async fn get_always_skills_async(&self) -> Vec<String> {
        self.get_always_skills()
    }

    /// 检查技能依赖是否满足
    pub fn check_requirements(&self, name: &str) -> RequirementCheck {
        self.controller.check_requirements(name)
    }

    /// 检查技能依赖是否满足（异步版本）
    pub async fn check_requirements_async(&self, name: &str) -> RequirementCheck {
        self.check_requirements(name)
    }

    /// 获取技能的安装指令
    pub fn get_install_instructions(&self, name: &str) -> Option<String> {
        self.controller.get_install_instructions(name)
    }

    /// 获取缺失依赖的描述字符串
    pub fn get_missing_requirements_description(&self, name: &str) -> String {
        self.controller.get_missing_requirements_description(name)
    }

    /// 构建技能摘要 XML 格式
    pub async fn build_skills_summary(&self) -> String {
        let all_skills = self.all_skill_entries();

        if all_skills.is_empty() {
            return String::new();
        }

        let mut lines = vec!["<skills>".to_string()];

        for entry in all_skills {
            let name = escape_xml(&entry.metadata.name);
            let desc = escape_xml(&entry.metadata.description);
            let path = entry.path.display().to_string();
            let check = self.check_requirements_async(&entry.metadata.name).await;
            let available = check.satisfied;

            lines.push(format!(
                "  <skill available=\"{}\" source=\"{}\">",
                available,
                escape_xml(&entry.source)
            ));
            lines.push(format!("    <name>{}</name>", name));
            lines.push(format!("    <description>{}</description>", desc));
            lines.push(format!("    <location>{}</location>", escape_xml(&path)));

            if !available {
                let missing = check
                    .missing
                    .iter()
                    .map(|r| match r {
                        super::Requirement::CliTool(t) => format!("CLI: {}", t),
                        super::Requirement::EnvVar(v) => format!("ENV: {}", v),
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                if !missing.is_empty() {
                    lines.push(format!("    <requires>{}</requires>", escape_xml(&missing)));
                }
            }

            lines.push("  </skill>".to_string());
        }

        lines.push("</skills>".to_string());

        lines.join("\n")
    }

    /// 获取技能描述
    pub async fn get_skill_description(&self, name: &str) -> String {
        self.get_all_metadata()
            .iter()
            .find(|m| m.name == name)
            .map(|m| {
                if m.description.is_empty() {
                    name.to_string()
                } else {
                    m.description.clone()
                }
            })
            .unwrap_or_else(|| name.to_string())
    }

    /// 获取所有 Skills 的元数据
    pub fn get_all_metadata(&self) -> Vec<SkillMetadata> {
        self.all_skill_entries()
            .into_iter()
            .map(|entry| entry.metadata)
            .collect()
    }

    /// 搜索相关 Skills
    pub fn search(&self, query: &str) -> Vec<String> {
        let query_lower = query.to_lowercase();
        self.get_all_metadata()
            .into_iter()
            .filter(|metadata| {
                metadata.name.to_lowercase().contains(&query_lower)
                    || metadata.description.to_lowercase().contains(&query_lower)
                    || metadata
                        .tags
                        .iter()
                        .any(|tag| tag.to_lowercase().contains(&query_lower))
            })
            .map(|metadata| metadata.name)
            .collect()
    }

    /// 检查 Skill 是否存在
    pub fn has_skill(&self, name: &str) -> bool {
        self.controller.has_skill(name) || self.managed_skill_markdown_path(name).is_some()
    }

    /// 重新扫描 Skills 目录
    pub fn rescan(&mut self) -> anyhow::Result<usize> {
        self.controller.scan_metadata()
    }

    /// 重新扫描 Skills 目录（异步版本）
    pub async fn rescan_async(&mut self) -> anyhow::Result<usize> {
        self.rescan()
    }

    /// 列出所有可用的技能信息
    pub async fn list_skills(&self, filter_unavailable: bool) -> Vec<SkillInfo> {
        let mut skills = Vec::new();

        for entry in self.all_skill_entries() {
            if filter_unavailable {
                let check = self.check_requirements_async(&entry.metadata.name).await;
                if !check.satisfied {
                    continue;
                }
            }

            skills.push(SkillInfo {
                name: entry.metadata.name.clone(),
                path: entry.path.display().to_string(),
                source: entry.source.clone(),
            });
        }

        skills
    }

    /// 同步本地 Hub 目录缓存
    pub async fn sync_hub_catalog(&self) -> anyhow::Result<Vec<HubSkillCatalogEntry>> {
        self.require_hub()?.sync_catalog().await
    }

    /// 搜索远程 Hub Skills
    pub async fn search_hub_skills(
        &self,
        query: &str,
        category: Option<&str>,
    ) -> anyhow::Result<Vec<HubSkillCatalogEntry>> {
        self.require_hub()?
            .search_with_category(query, category)
            .await
    }

    /// 列出 Hub 分类
    pub async fn list_hub_categories(&self) -> anyhow::Result<Vec<String>> {
        self.require_hub()?.categories().await
    }

    /// 获取 Hub Skill 详情
    pub async fn hub_skill_details(
        &self,
        name: &str,
    ) -> anyhow::Result<Option<HubSkillCatalogEntry>> {
        self.require_hub()?.skill_details(name).await
    }

    /// 安装 Hub Skill
    pub async fn install_hub_skill(
        &self,
        name: &str,
        version: Option<&str>,
    ) -> anyhow::Result<ManagedHubSkillRecord> {
        self.require_hub()?.install(name, version).await
    }

    /// 更新单个 Hub Skill
    pub async fn update_hub_skill(&self, name: &str) -> anyhow::Result<ManagedHubSkillRecord> {
        self.require_hub()?.update(name).await
    }

    /// 更新全部 Hub Skills
    pub async fn update_all_hub_skills(&self) -> anyhow::Result<Vec<ManagedHubSkillRecord>> {
        self.require_hub()?.update_all().await
    }

    /// 删除 Hub Skill
    pub fn remove_hub_skill(&self, name: &str) -> anyhow::Result<bool> {
        self.require_hub()?.remove(name)
    }

    /// 列出已安装 Hub Skills
    pub fn list_hub_installed_skills(&self) -> anyhow::Result<Vec<ManagedHubSkillRecord>> {
        self.require_hub()?.list_installed()
    }

    /// 获取 Hub 缓存状态
    pub fn hub_cache_status(&self) -> anyhow::Result<HubCacheStatus> {
        self.require_hub()?.cache_status()
    }

    /// 清理 Hub 缓存
    pub fn clear_hub_cache(&self) -> anyhow::Result<bool> {
        self.require_hub()?.clear_cache()
    }

    fn require_hub(&self) -> anyhow::Result<&SkillHubClient> {
        self.hub
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("skills hub is not configured for this manager"))
    }

    fn local_skill_markdown_path(&self, name: &str) -> Option<PathBuf> {
        let skill_path = self.controller.get_skill_path(name)?;
        let skill_md = skill_path.join("SKILL.md");
        skill_md.exists().then_some(skill_md)
    }

    fn managed_skill_markdown_path(&self, name: &str) -> Option<PathBuf> {
        let hub = self.hub.as_ref()?;
        let skill_md = hub.managed_skill_path(name).ok()?.join("SKILL.md");
        skill_md.exists().then_some(skill_md)
    }

    fn all_skill_entries(&self) -> Vec<SkillEntry> {
        let mut entries = Vec::new();
        let mut seen = HashSet::new();

        for metadata in self.controller.get_all_metadata() {
            let path = self
                .controller
                .get_skill_path(&metadata.name)
                .unwrap_or_else(|| PathBuf::from(&metadata.name));
            seen.insert(metadata.name.clone());
            entries.push(SkillEntry {
                metadata,
                path,
                source: "skills".to_string(),
            });
        }

        for hub_entry in self.hub_installed_entries() {
            if seen.insert(hub_entry.metadata.name.clone()) {
                entries.push(hub_entry);
            }
        }

        entries.sort_by(|left, right| left.metadata.name.cmp(&right.metadata.name));
        entries
    }

    fn hub_installed_entries(&self) -> Vec<SkillEntry> {
        let Some(hub) = &self.hub else {
            return Vec::new();
        };

        let records = match hub.list_installed() {
            Ok(records) => records,
            Err(_) => return Vec::new(),
        };

        records
            .into_iter()
            .map(|record| {
                let skill_dir = hub
                    .managed_skill_path(&record.name)
                    .unwrap_or_else(|_| PathBuf::from(&record.name));
                let skill_md = skill_dir.join("SKILL.md");
                let metadata = if skill_md.exists() {
                    SkillParser::parse_from_file(&skill_md)
                        .map(|(metadata, _)| metadata)
                        .unwrap_or_else(|_| metadata_from_record(&record))
                } else {
                    metadata_from_record(&record)
                };

                SkillEntry {
                    metadata,
                    path: skill_dir,
                    source: "hub".to_string(),
                }
            })
            .collect()
    }
}

#[async_trait::async_trait]
impl PromptSkillsManager for SkillsManager {
    async fn get_always_skills(&self) -> Vec<String> {
        SkillsManager::get_always_skills(self)
    }

    async fn load_skills_for_context(&self, names: &[String]) -> String {
        SkillsManager::load_skills_for_context(self, names).await
    }

    async fn build_skills_summary(&self) -> String {
        SkillsManager::build_skills_summary(self).await
    }
}

/// 技能信息
#[derive(Debug, Clone)]
pub struct SkillInfo {
    pub name: String,
    pub path: String,
    pub source: String,
}

#[derive(Debug, Clone)]
struct SkillEntry {
    metadata: SkillMetadata,
    path: PathBuf,
    source: String,
}

fn metadata_from_record(record: &ManagedHubSkillRecord) -> SkillMetadata {
    SkillMetadata {
        name: record.name.clone(),
        description: record.description.clone(),
        category: record.categories.first().cloned(),
        tags: Vec::new(),
        version: record.version.clone(),
        author: None,
        always: false,
        requires: Some(SkillRequirements::default()),
        install: None,
    }
}

fn read_skill_markdown(skill_md: &Path) -> Option<String> {
    let content = std::fs::read_to_string(skill_md).ok()?;
    strip_frontmatter(content)
}

async fn read_skill_markdown_async(skill_md: &Path) -> Option<String> {
    let content = tokio::fs::read_to_string(skill_md).await.ok()?;
    strip_frontmatter(content)
}

fn strip_frontmatter(content: String) -> Option<String> {
    let parts: Vec<&str> = content.splitn(3, "---").collect();
    if parts.len() >= 3 {
        Some(parts[2].trim().to_string())
    } else {
        Some(content)
    }
}

/// Escape XML 特殊字符
fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::context::{AgentContextBuilder, AgentIdentity};
    use axum::{Json, Router, routing::get};
    use serde_json::json;
    use std::fs;
    use std::sync::Arc;
    use tempfile::TempDir;
    use tokio::net::TcpListener;

    fn create_test_skill(dir: &Path, name: &str, description: &str) -> std::io::Result<()> {
        let skill_dir = dir.join(name);
        fs::create_dir_all(&skill_dir)?;

        let content = format!(
            r#"---
name: {}
description: {}
category: test
tags: [test]
version: "1.0.0"
---

# {} Skill

This is a test skill."#,
            name, description, name
        );

        fs::write(skill_dir.join("SKILL.md"), content)?;
        Ok(())
    }

    fn hub_config(
        catalog_url: impl Into<String>,
        temp: &TempDir,
        auto_install_on_miss: bool,
    ) -> SkillHubClientConfig {
        SkillHubClientConfig::new(catalog_url)
            .with_managed_root(temp.path().join("managed"))
            .with_cache_root(temp.path().join("cache"))
            .with_auto_install_on_miss(auto_install_on_miss)
    }

    #[test]
    fn test_new_manager() {
        let temp_dir = TempDir::new().unwrap();
        let skills_dir = temp_dir.path();

        create_test_skill(skills_dir, "skill1", "First skill").unwrap();
        create_test_skill(skills_dir, "skill2", "Second skill").unwrap();

        let manager = SkillsManager::new(skills_dir).unwrap();
        assert_eq!(manager.get_all_metadata().len(), 2);
    }

    #[test]
    fn test_new_manager_nonexistent_dir() {
        let result = SkillsManager::new("/nonexistent/skills");
        assert!(result.is_ok());
        assert!(result.unwrap().get_all_metadata().is_empty());
    }

    #[test]
    fn test_build_system_prompt() {
        let temp_dir = TempDir::new().unwrap();
        let skills_dir = temp_dir.path();

        create_test_skill(skills_dir, "skill1", "First skill").unwrap();

        let manager = SkillsManager::new(skills_dir).unwrap();
        let prompt = manager.build_system_prompt();
        assert!(prompt.contains("skill1"));
        assert!(prompt.contains("First skill"));
    }

    #[test]
    fn test_load_skill() {
        let temp_dir = TempDir::new().unwrap();
        let skills_dir = temp_dir.path();

        create_test_skill(skills_dir, "skill1", "First skill").unwrap();

        let manager = SkillsManager::new(skills_dir).unwrap();
        let content = manager.load_skill("skill1").unwrap();
        assert!(content.contains("# skill1 Skill"));
        assert!(content.contains("This is a test skill"));
        assert!(!content.contains("name:"));
    }

    #[test]
    fn test_search() {
        let temp_dir = TempDir::new().unwrap();
        let skills_dir = temp_dir.path();

        create_test_skill(skills_dir, "pdf_processing", "Process PDF files").unwrap();
        create_test_skill(skills_dir, "web_scraping", "Scrape web pages").unwrap();

        let manager = SkillsManager::new(skills_dir).unwrap();

        let results = manager.search("pdf");
        assert_eq!(results, vec!["pdf_processing".to_string()]);

        let results = manager.search("web");
        assert_eq!(results, vec!["web_scraping".to_string()]);
    }

    #[test]
    fn test_has_skill() {
        let temp_dir = TempDir::new().unwrap();
        let skills_dir = temp_dir.path();

        create_test_skill(skills_dir, "skill1", "First skill").unwrap();

        let manager = SkillsManager::new(skills_dir).unwrap();
        assert!(manager.has_skill("skill1"));
        assert!(!manager.has_skill("skill2"));
    }

    #[tokio::test]
    async fn test_manager_auto_installs_skill_on_async_load() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = Router::new()
            .route(
                "/catalog",
                get(move || async move {
                    Json(json!({
                        "skills": [
                            {
                                "name": "pdf_processing",
                                "description": "Process PDF files",
                                "category": "documents",
                                "compatibility": ["mofa"],
                                "latestVersion": "1.2.0",
                                "downloadUrl": format!("http://{address}/bundle")
                            }
                        ]
                    }))
                }),
            )
            .route(
                "/bundle",
                get(|| async {
                    Json(json!({
                        "version": "1.2.0",
                        "files": [
                            {
                                "path": "SKILL.md",
                                "content": "---\nname: pdf_processing\ndescription: Process PDF files\n---\n# PDF\n\nAuto-installed"
                            }
                        ]
                    }))
                }),
            );
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let temp_dir = TempDir::new().unwrap();
        let manager = SkillsManager::with_hub(
            temp_dir.path(),
            hub_config(format!("http://{address}/catalog"), &temp_dir, true),
        )
        .unwrap();

        let content = manager.load_skill_async("pdf_processing").await.unwrap();
        assert!(content.contains("Auto-installed"));

        let listed = manager.list_skills(false).await;
        assert!(
            listed
                .iter()
                .any(|skill| skill.name == "pdf_processing" && skill.source == "hub")
        );

        server.abort();
    }

    #[tokio::test]
    async fn test_manager_integrates_with_agent_context_builder() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = Router::new()
            .route(
                "/catalog",
                get(move || async move {
                    Json(json!({
                        "skills": [
                            {
                                "name": "pdf_processing",
                                "description": "Process PDF files",
                                "category": "documents",
                                "compatibility": ["mofa"],
                                "latestVersion": "1.2.0",
                                "downloadUrl": format!("http://{address}/bundle")
                            }
                        ]
                    }))
                }),
            )
            .route(
                "/bundle",
                get(|| async {
                    Json(json!({
                        "version": "1.2.0",
                        "files": [
                            {
                                "path": "SKILL.md",
                                "content": "---\nname: pdf_processing\ndescription: Process PDF files\n---\n# PDF\n\nInjected"
                            }
                        ]
                    }))
                }),
            );
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let temp_dir = TempDir::new().unwrap();
        let manager = Arc::new(
            SkillsManager::with_hub(
                temp_dir.path(),
                hub_config(format!("http://{address}/catalog"), &temp_dir, true),
            )
            .unwrap(),
        );

        let builder = AgentContextBuilder::new(temp_dir.path().to_path_buf())
            .with_identity(AgentIdentity::new("agent", "Hub-aware"))
            .with_skills(manager);
        let requested = vec!["pdf_processing".to_string()];
        let messages = builder
            .build_messages_with_skills(Vec::new(), "hello", None, Some(&requested))
            .await
            .unwrap();

        let system = format!("{:?}", messages[0]);
        assert!(system.contains("Requested Skills"));
        assert!(system.contains("Injected"));

        server.abort();
    }
}
