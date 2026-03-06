//! Agent Skills 管理 API
//!
//! 提供 Skills 的统一管理接口，支持：
//! - 渐进式披露（Progressive Disclosure）
//! - 热更新支持
//! - 搜索和加载

pub mod hub;
pub mod manager;

pub use hub::{
    DEFAULT_SKILLS_HUB_CATALOG_URL, HubCacheStatus, HubCatalogCache, HubSkillBundle,
    HubSkillBundleFile, HubSkillCatalogEntry, HubSkillVersion, InstalledHubSkill,
    ManagedHubSkillRecord, SkillHubClient, SkillHubClientConfig, default_managed_skills_root,
    default_skills_cache_root, install_skill_bundle,
};
pub use manager::{SkillInfo, SkillsManager};

// 重新导出 skill 相关类型
pub use mofa_plugins::skill::{
    DisclosureController, Requirement, RequirementCheck, SkillMetadata, SkillParser,
    SkillRequirements, SkillState, SkillVersion,
};

use std::path::PathBuf;

/// Skills 管理器构建器
#[derive(Debug, Clone)]
pub struct SkillsManagerBuilder {
    search_dirs: Vec<PathBuf>,
    hub_config: Option<SkillHubClientConfig>,
}

impl SkillsManagerBuilder {
    /// 创建新的构建器
    pub fn new(skills_dir: impl Into<PathBuf>) -> Self {
        Self {
            search_dirs: vec![skills_dir.into()],
            hub_config: None,
        }
    }

    /// 设置 skills 目录（单目录）
    pub fn with_skills_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.search_dirs = vec![dir.into()];
        self
    }

    /// 添加搜索目录（多目录，按优先级排序）
    pub fn with_search_dirs(mut self, dirs: Vec<PathBuf>) -> Self {
        self.search_dirs = dirs;
        self
    }

    /// 添加一个搜索目录
    pub fn add_search_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.search_dirs.push(dir.into());
        self
    }

    /// 启用 Skills Hub
    pub fn with_hub(mut self, config: SkillHubClientConfig) -> Self {
        self.hub_config = Some(config);
        self
    }

    /// 构建 SkillsManager
    pub fn build(&self) -> anyhow::Result<SkillsManager> {
        self.build_multi()
    }

    /// 构建支持多目录的 SkillsManager
    pub fn build_multi(&self) -> anyhow::Result<SkillsManager> {
        if let Some(config) = &self.hub_config {
            SkillsManager::with_search_dirs_and_hub(self.search_dirs.clone(), config.clone())
        } else if self.search_dirs.len() == 1 {
            SkillsManager::new(&self.search_dirs[0])
        } else {
            SkillsManager::with_search_dirs(self.search_dirs.clone())
        }
    }
}

/// 便捷函数：从目录创建 SkillsManager
pub fn from_dir(skills_dir: impl AsRef<std::path::Path>) -> anyhow::Result<SkillsManager> {
    SkillsManager::new(skills_dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn write_skill(root: &std::path::Path, name: &str) {
        let skill_dir = root.join(name);
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            format!("---\nname: {name}\ndescription: Test skill\n---\n# {name}"),
        )
        .unwrap();
    }

    #[test]
    fn test_builder_build_respects_multiple_search_dirs() {
        let temp = TempDir::new().unwrap();
        let primary = temp.path().join("primary");
        let secondary = temp.path().join("secondary");
        std::fs::create_dir_all(&primary).unwrap();
        std::fs::create_dir_all(&secondary).unwrap();
        write_skill(&secondary, "pdf_processing");

        let manager = SkillsManagerBuilder::new(&primary)
            .with_search_dirs(vec![primary.clone(), secondary.clone()])
            .build()
            .unwrap();

        assert!(manager.load_skill("pdf_processing").is_some());
    }
}
