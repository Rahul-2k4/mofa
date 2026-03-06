use anyhow::{Context, Result, bail};
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fs;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const DEFAULT_SKILLS_HUB_CATALOG_URL: &str = "https://clawhub.run/api/skills/catalog";
const DEFAULT_COMPATIBILITY_TARGETS: &[&str] = &["mofa", "universal", "cross-platform", "portable"];
const CATALOG_CACHE_FILE: &str = "catalog.json";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HubSkillVersion {
    pub version: String,
    #[serde(default, alias = "downloadUrl")]
    pub download_url: Option<String>,
    #[serde(default)]
    pub rating: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HubSkillCatalogEntry {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub author: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub category: Option<String>,
    #[serde(default)]
    pub categories: Vec<String>,
    #[serde(default)]
    pub compatibility: Vec<String>,
    #[serde(default, alias = "latestVersion")]
    pub latest_version: Option<String>,
    #[serde(default, alias = "downloadUrl")]
    pub download_url: Option<String>,
    #[serde(default)]
    pub versions: Vec<HubSkillVersion>,
    #[serde(default)]
    pub rating: Option<f32>,
    #[serde(default)]
    pub downloads: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HubSkillBundleFile {
    pub path: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HubSkillBundle {
    #[serde(default)]
    pub version: Option<String>,
    pub files: Vec<HubSkillBundleFile>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InstalledHubSkill {
    pub name: String,
    pub version: Option<String>,
    pub install_dir: PathBuf,
    pub source_url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ManagedHubSkillRecord {
    pub name: String,
    pub description: String,
    pub version: Option<String>,
    pub source: String,
    pub source_url: Option<String>,
    pub installed_path: PathBuf,
    pub catalog_url: String,
    #[serde(default)]
    pub categories: Vec<String>,
    #[serde(default)]
    pub compatibility: Vec<String>,
    #[serde(default)]
    pub rating: Option<f32>,
    pub installed_at_epoch_secs: u64,
    #[serde(default)]
    pub updated_at_epoch_secs: Option<u64>,
    #[serde(default)]
    pub last_used_at_epoch_secs: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HubCatalogCache {
    pub catalog_url: String,
    pub fetched_at_epoch_secs: u64,
    pub entries: Vec<HubSkillCatalogEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HubCacheStatus {
    pub path: PathBuf,
    pub exists: bool,
    pub entry_count: usize,
    pub fetched_at_epoch_secs: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct SkillHubClientConfig {
    pub catalog_url: String,
    pub auth_token: Option<String>,
    pub managed_root: PathBuf,
    pub cache_root: PathBuf,
    pub auto_install_on_miss: bool,
    pub compatibility_targets: Vec<String>,
}

impl SkillHubClientConfig {
    fn defaults() -> Self {
        Self {
            catalog_url: DEFAULT_SKILLS_HUB_CATALOG_URL.to_string(),
            auth_token: Self::env_auth_token(),
            managed_root: default_managed_skills_root()
                .unwrap_or_else(|_| PathBuf::from(".mofa/skills/hub")),
            cache_root: default_skills_cache_root()
                .unwrap_or_else(|_| PathBuf::from(".mofa/cache/skills-hub")),
            auto_install_on_miss: false,
            compatibility_targets: DEFAULT_COMPATIBILITY_TARGETS
                .iter()
                .map(|target| target.to_string())
                .collect(),
        }
    }

    pub fn new(catalog_url: impl Into<String>) -> Self {
        let mut config = Self::defaults();
        config.catalog_url = catalog_url.into();
        config
    }

    pub fn from_env() -> Self {
        let mut config = Self::defaults();
        if let Ok(catalog_url) = std::env::var("MOFA_SKILLS_HUB_CATALOG_URL") {
            config.catalog_url = catalog_url;
        }
        if let Some(auto_install) = read_bool_env("MOFA_SKILLS_HUB_AUTO_INSTALL") {
            config.auto_install_on_miss = auto_install;
        }
        config
    }

    pub fn with_auth_token(mut self, auth_token: impl Into<String>) -> Self {
        self.auth_token = Some(auth_token.into());
        self
    }

    pub fn with_managed_root(mut self, managed_root: impl Into<PathBuf>) -> Self {
        self.managed_root = managed_root.into();
        self
    }

    pub fn with_cache_root(mut self, cache_root: impl Into<PathBuf>) -> Self {
        self.cache_root = cache_root.into();
        self
    }

    pub fn with_auto_install_on_miss(mut self, enabled: bool) -> Self {
        self.auto_install_on_miss = enabled;
        self
    }

    pub fn with_compatibility_targets(mut self, targets: Vec<String>) -> Self {
        self.compatibility_targets = targets;
        self
    }

    fn env_auth_token() -> Option<String> {
        std::env::var("MOFA_SKILLS_HUB_TOKEN")
            .ok()
            .or_else(|| std::env::var("CLAWHUB_AUTH_TOKEN").ok())
            .or_else(|| std::env::var("CLAWHUB_API_KEY").ok())
    }
}

#[derive(Debug, Clone)]
pub struct SkillHubClient {
    client: reqwest::Client,
    config: SkillHubClientConfig,
}

impl SkillHubClient {
    pub fn new(config: SkillHubClientConfig) -> Result<Self> {
        fs::create_dir_all(&config.managed_root).with_context(|| {
            format!(
                "failed to create managed root {}",
                config.managed_root.display()
            )
        })?;
        fs::create_dir_all(&config.cache_root).with_context(|| {
            format!(
                "failed to create cache root {}",
                config.cache_root.display()
            )
        })?;
        fs::create_dir_all(registry_dir(&config)).with_context(|| {
            format!(
                "failed to create registry dir {}",
                registry_dir(&config).display()
            )
        })?;

        let mut headers = HeaderMap::new();
        if let Some(token) = &config.auth_token {
            let value = HeaderValue::from_str(&format!("Bearer {}", token))
                .context("invalid hub auth token header")?;
            headers.insert(AUTHORIZATION, value);
        }

        let client = reqwest::Client::builder()
            .default_headers(headers)
            .build()
            .context("failed to build hub HTTP client")?;

        Ok(Self { client, config })
    }

    pub fn config(&self) -> &SkillHubClientConfig {
        &self.config
    }

    pub async fn fetch_catalog(&self) -> Result<Vec<HubSkillCatalogEntry>> {
        self.fetch_catalog_remote().await
    }

    pub async fn sync_catalog(&self) -> Result<Vec<HubSkillCatalogEntry>> {
        match self.fetch_catalog_remote().await {
            Ok(entries) => {
                self.write_catalog_cache(&entries)?;
                Ok(entries)
            }
            Err(remote_error) => self.load_cached_catalog().with_context(|| {
                format!("remote sync failed and cache unavailable: {remote_error:#}")
            }),
        }
    }

    pub fn load_cached_catalog(&self) -> Result<Vec<HubSkillCatalogEntry>> {
        let cache_path = self.catalog_cache_path();
        let bytes = fs::read(&cache_path)
            .with_context(|| format!("failed to read catalog cache {}", cache_path.display()))?;
        let cache: HubCatalogCache =
            serde_json::from_slice(&bytes).context("failed to decode catalog cache")?;
        if cache.catalog_url != self.config.catalog_url {
            bail!(
                "catalog cache for '{}' does not match configured hub '{}'",
                cache.catalog_url,
                self.config.catalog_url
            );
        }
        Ok(cache.entries)
    }

    pub fn cache_status(&self) -> Result<HubCacheStatus> {
        let path = self.catalog_cache_path();
        if !path.exists() {
            return Ok(HubCacheStatus {
                path,
                exists: false,
                entry_count: 0,
                fetched_at_epoch_secs: None,
            });
        }

        let bytes = fs::read(&path)
            .with_context(|| format!("failed to read catalog cache {}", path.display()))?;
        let cache: HubCatalogCache =
            serde_json::from_slice(&bytes).context("failed to decode catalog cache")?;

        Ok(HubCacheStatus {
            path,
            exists: true,
            entry_count: cache.entries.len(),
            fetched_at_epoch_secs: Some(cache.fetched_at_epoch_secs),
        })
    }

    pub fn clear_cache(&self) -> Result<bool> {
        let path = self.catalog_cache_path();
        if !path.exists() {
            return Ok(false);
        }

        fs::remove_file(&path)
            .with_context(|| format!("failed to remove cache file {}", path.display()))?;
        Ok(true)
    }

    pub async fn search(&self, query: &str) -> Result<Vec<HubSkillCatalogEntry>> {
        self.search_with_category(query, None).await
    }

    pub async fn search_with_category(
        &self,
        query: &str,
        category: Option<&str>,
    ) -> Result<Vec<HubSkillCatalogEntry>> {
        let query = query.trim().to_lowercase();
        let category = category.map(|value| value.trim().to_lowercase());
        let results = self
            .sync_catalog()
            .await?
            .into_iter()
            .filter(|entry| {
                let query_match = query.is_empty() || entry.matches(&query);
                let category_match = category.as_ref().map_or(true, |selected| {
                    entry.categories().iter().any(|item| item == selected)
                });
                query_match && category_match
            })
            .collect();
        Ok(results)
    }

    pub async fn categories(&self) -> Result<Vec<String>> {
        let mut categories = BTreeSet::new();
        for entry in self.sync_catalog().await? {
            for category in entry.categories() {
                categories.insert(category);
            }
        }
        Ok(categories.into_iter().collect())
    }

    pub async fn skill_details(&self, name: &str) -> Result<Option<HubSkillCatalogEntry>> {
        Ok(self
            .sync_catalog()
            .await?
            .into_iter()
            .find(|entry| entry.name == name))
    }

    pub async fn install_from_catalog_entry(
        &self,
        install_root: impl AsRef<Path>,
        entry: &HubSkillCatalogEntry,
        version: Option<&str>,
    ) -> Result<InstalledHubSkill> {
        self.validate_compatibility(entry)?;
        validate_skill_name(&entry.name)?;
        let download_url = entry
            .download_url_for(version)
            .with_context(|| format!("no download URL available for skill {}", entry.name))?;

        let response = self
            .client
            .get(&download_url)
            .send()
            .await
            .with_context(|| format!("failed to download skill bundle from {}", download_url))?;
        let response = response
            .error_for_status()
            .with_context(|| format!("hub bundle download failed for {}", download_url))?;
        let bundle: HubSkillBundle = response
            .json()
            .await
            .context("failed to decode hub bundle response")?;

        install_skill_bundle(install_root, &entry.name, &bundle, Some(&download_url))
    }

    pub async fn install(
        &self,
        name: &str,
        version: Option<&str>,
    ) -> Result<ManagedHubSkillRecord> {
        validate_skill_name(name)?;
        let entry = self
            .skill_details(name)
            .await?
            .with_context(|| format!("skill '{name}' not found in remote catalog"))?;
        let previous = self.get_installed(&entry.name)?;
        let installed = self
            .install_from_catalog_entry(&self.config.managed_root, &entry, version)
            .await?;
        let now = now_epoch_secs();

        let record = ManagedHubSkillRecord {
            name: entry.name.clone(),
            description: entry.description.clone(),
            version: installed.version.or_else(|| entry.latest_version.clone()),
            source: "hub".to_string(),
            source_url: installed.source_url.clone(),
            installed_path: installed.install_dir.clone(),
            catalog_url: self.config.catalog_url.clone(),
            categories: entry.categories().into_iter().collect(),
            compatibility: entry.compatibility.clone(),
            rating: entry.rating,
            installed_at_epoch_secs: previous
                .as_ref()
                .map(|record| record.installed_at_epoch_secs)
                .unwrap_or(now),
            updated_at_epoch_secs: previous.as_ref().map(|_| now),
            last_used_at_epoch_secs: Some(now),
        };

        self.save_record(&record)?;
        Ok(record)
    }

    pub async fn update(&self, name: &str) -> Result<ManagedHubSkillRecord> {
        validate_skill_name(name)?;
        let installed = self
            .get_installed(name)?
            .with_context(|| format!("skill '{name}' is not installed"))?;
        let entry = self
            .skill_details(name)
            .await?
            .with_context(|| format!("skill '{name}' not found in remote catalog"))?;

        let target_version = entry
            .latest_version
            .clone()
            .or_else(|| installed.version.clone());

        if installed.version == target_version {
            self.touch_last_used(name)?;
            return Ok(installed);
        }

        self.install(name, target_version.as_deref()).await
    }

    pub async fn update_all(&self) -> Result<Vec<ManagedHubSkillRecord>> {
        let installed = self.list_installed()?;
        let mut updated = Vec::new();
        for record in installed {
            updated.push(self.update(&record.name).await?);
        }
        Ok(updated)
    }

    pub fn remove(&self, name: &str) -> Result<bool> {
        let mut removed = false;
        let install_dir = self.managed_skill_path(name)?;
        if install_dir.exists() {
            fs::remove_dir_all(&install_dir).with_context(|| {
                format!("failed to remove installed skill {}", install_dir.display())
            })?;
            removed = true;
        }

        let record_path = record_path(&self.config, name);
        if record_path.exists() {
            fs::remove_file(&record_path).with_context(|| {
                format!(
                    "failed to remove skill registry entry {}",
                    record_path.display()
                )
            })?;
            removed = true;
        }

        Ok(removed)
    }

    pub fn list_installed(&self) -> Result<Vec<ManagedHubSkillRecord>> {
        let mut records = Vec::new();
        let dir = registry_dir(&self.config);
        if !dir.exists() {
            return Ok(records);
        }

        for entry in fs::read_dir(&dir)
            .with_context(|| format!("failed to read registry dir {}", dir.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
                continue;
            }
            let bytes = fs::read(&path)
                .with_context(|| format!("failed to read registry entry {}", path.display()))?;
            let record: ManagedHubSkillRecord =
                serde_json::from_slice(&bytes).context("failed to decode registry entry")?;
            records.push(record);
        }

        records.sort_by(|left, right| left.name.cmp(&right.name));
        Ok(records)
    }

    pub fn get_installed(&self, name: &str) -> Result<Option<ManagedHubSkillRecord>> {
        validate_skill_name(name)?;
        let path = record_path(&self.config, name);
        if !path.exists() {
            return Ok(None);
        }

        let bytes = fs::read(&path)
            .with_context(|| format!("failed to read registry entry {}", path.display()))?;
        let record: ManagedHubSkillRecord =
            serde_json::from_slice(&bytes).context("failed to decode registry entry")?;
        Ok(Some(record))
    }

    pub fn touch_last_used(&self, name: &str) -> Result<()> {
        if let Some(mut record) = self.get_installed(name)? {
            record.last_used_at_epoch_secs = Some(now_epoch_secs());
            self.save_record(&record)?;
        }
        Ok(())
    }

    pub async fn ensure_installed(&self, name: &str) -> Result<Option<ManagedHubSkillRecord>> {
        validate_skill_name(name)?;
        if let Some(record) = self.get_installed(name)? {
            self.touch_last_used(name)?;
            return Ok(Some(record));
        }

        if !self.config.auto_install_on_miss {
            return Ok(None);
        }

        if self.skill_details(name).await?.is_none() {
            return Ok(None);
        }

        let record = self.install(name, None).await?;
        Ok(Some(record))
    }

    pub fn managed_skill_path(&self, name: &str) -> Result<PathBuf> {
        managed_skill_path(&self.config.managed_root, name)
    }

    pub fn is_managed_skill_path(&self, path: &Path) -> bool {
        path.starts_with(&self.config.managed_root)
    }

    fn validate_compatibility(&self, entry: &HubSkillCatalogEntry) -> Result<()> {
        if entry.compatibility.is_empty() {
            return Ok(());
        }

        let targets: BTreeSet<String> = self
            .config
            .compatibility_targets
            .iter()
            .map(|target| normalize_compatibility_label(target))
            .filter(|target| !target.is_empty())
            .collect();
        if targets.is_empty() {
            return Ok(());
        }

        let compatible = entry
            .compatibility
            .iter()
            .map(|declared| normalize_compatibility_label(declared))
            .filter(|declared| !declared.is_empty())
            .any(|declared| targets.contains(&declared));

        if compatible {
            Ok(())
        } else {
            bail!(
                "skill '{}' is not compatible with MoFA targets: {:?}",
                entry.name,
                self.config.compatibility_targets
            )
        }
    }

    fn catalog_cache_path(&self) -> PathBuf {
        self.config.cache_root.join(format!(
            "{}-{}.json",
            CATALOG_CACHE_FILE.trim_end_matches(".json"),
            cache_key(&self.config.catalog_url)
        ))
    }

    fn save_record(&self, record: &ManagedHubSkillRecord) -> Result<()> {
        let path = record_path(&self.config, &record.name);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create registry parent {}", parent.display())
            })?;
        }
        let payload = serde_json::to_vec_pretty(record)?;
        fs::write(&path, payload)
            .with_context(|| format!("failed to write registry entry {}", path.display()))?;
        Ok(())
    }

    fn write_catalog_cache(&self, entries: &[HubSkillCatalogEntry]) -> Result<()> {
        let cache = HubCatalogCache {
            catalog_url: self.config.catalog_url.clone(),
            fetched_at_epoch_secs: now_epoch_secs(),
            entries: entries.to_vec(),
        };
        let payload = serde_json::to_vec_pretty(&cache)?;
        let path = self.catalog_cache_path();
        fs::create_dir_all(&self.config.cache_root).with_context(|| {
            format!(
                "failed to create cache root {}",
                self.config.cache_root.display()
            )
        })?;
        fs::write(&path, payload)
            .with_context(|| format!("failed to write catalog cache {}", path.display()))?;
        Ok(())
    }

    async fn fetch_catalog_remote(&self) -> Result<Vec<HubSkillCatalogEntry>> {
        let response = self
            .client
            .get(&self.config.catalog_url)
            .send()
            .await
            .with_context(|| {
                format!(
                    "failed to fetch skill catalog from {}",
                    self.config.catalog_url
                )
            })?;
        let response = response.error_for_status().with_context(|| {
            format!("hub catalog request failed for {}", self.config.catalog_url)
        })?;
        let payload: CatalogResponse = response
            .json()
            .await
            .context("failed to decode hub catalog response")?;
        Ok(payload.into_entries())
    }
}

fn normalize_compatibility_label(value: &str) -> String {
    let mut normalized = value
        .trim()
        .to_ascii_lowercase()
        .replace([' ', '_'], "-")
        .trim_matches('-')
        .to_string();

    for suffix in ["-compatible", "-compat", "-skill", "-skills"] {
        if let Some(stripped) = normalized.strip_suffix(suffix) {
            normalized = stripped.trim_end_matches('-').to_string();
        }
    }

    normalized
}

impl HubSkillCatalogEntry {
    fn matches(&self, query: &str) -> bool {
        self.name.to_lowercase().contains(query)
            || self.description.to_lowercase().contains(query)
            || self
                .author
                .as_ref()
                .map(|author| author.to_lowercase().contains(query))
                .unwrap_or(false)
            || self
                .tags
                .iter()
                .any(|tag| tag.to_lowercase().contains(query))
            || self
                .categories()
                .iter()
                .any(|category| category.contains(query))
    }

    pub fn categories(&self) -> BTreeSet<String> {
        let mut categories = BTreeSet::new();
        if let Some(category) = &self.category {
            if !category.trim().is_empty() {
                categories.insert(category.trim().to_lowercase());
            }
        }
        for category in &self.categories {
            if !category.trim().is_empty() {
                categories.insert(category.trim().to_lowercase());
            }
        }
        categories
    }

    fn download_url_for(&self, version: Option<&str>) -> Option<String> {
        if let Some(version) = version {
            if let Some(found) = self
                .versions
                .iter()
                .find(|candidate| candidate.version == version)
                .and_then(|candidate| candidate.download_url.clone())
            {
                return Some(found);
            }
        }

        self.download_url.clone().or_else(|| {
            self.versions
                .iter()
                .find(|candidate| {
                    self.latest_version
                        .as_deref()
                        .map(|latest| candidate.version == latest)
                        .unwrap_or(false)
                })
                .and_then(|candidate| candidate.download_url.clone())
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum CatalogResponse {
    Entries(Vec<HubSkillCatalogEntry>),
    Wrapped { skills: Vec<HubSkillCatalogEntry> },
}

impl CatalogResponse {
    fn into_entries(self) -> Vec<HubSkillCatalogEntry> {
        match self {
            Self::Entries(entries) => entries,
            Self::Wrapped { skills } => skills,
        }
    }
}

pub fn install_skill_bundle(
    install_root: impl AsRef<Path>,
    skill_name: &str,
    bundle: &HubSkillBundle,
    source_url: Option<&str>,
) -> Result<InstalledHubSkill> {
    let install_root = install_root.as_ref();
    let install_dir = managed_skill_path(install_root, skill_name)?;
    let staged_files = validate_bundle_files(bundle)?;
    let stage_dir = unique_work_dir(install_root, skill_name, "staging");
    let backup_dir = unique_work_dir(install_root, skill_name, "backup");

    fs::create_dir_all(&stage_dir)
        .with_context(|| format!("failed to create staging directory {}", stage_dir.display()))?;

    for (file, relative_path) in bundle.files.iter().zip(staged_files.iter()) {
        let target_path = stage_dir.join(relative_path);
        if let Some(parent) = target_path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!(
                    "failed to create skill parent directory {}",
                    parent.display()
                )
            })?;
        }
        if let Err(error) = fs::write(&target_path, &file.content).with_context(|| {
            format!(
                "failed to write installed skill file {}",
                target_path.display()
            )
        }) {
            let _ = fs::remove_dir_all(&stage_dir);
            return Err(error);
        }
    }

    let had_existing_install = install_dir.exists();
    if had_existing_install {
        fs::rename(&install_dir, &backup_dir).with_context(|| {
            format!(
                "failed to move existing install {} to backup {}",
                install_dir.display(),
                backup_dir.display()
            )
        })?;
    }

    if let Err(error) = fs::rename(&stage_dir, &install_dir).with_context(|| {
        format!(
            "failed to promote staging directory {} to install {}",
            stage_dir.display(),
            install_dir.display()
        )
    }) {
        let _ = fs::remove_dir_all(&stage_dir);
        if had_existing_install {
            let _ = fs::rename(&backup_dir, &install_dir);
        }
        return Err(error);
    }

    if had_existing_install {
        fs::remove_dir_all(&backup_dir).with_context(|| {
            format!(
                "failed to remove backup install directory {}",
                backup_dir.display()
            )
        })?;
    }

    Ok(InstalledHubSkill {
        name: skill_name.to_string(),
        version: bundle.version.clone(),
        install_dir,
        source_url: source_url.map(ToOwned::to_owned),
    })
}

pub fn default_managed_skills_root() -> Result<PathBuf> {
    let data_dir = dirs_next::data_local_dir()
        .ok_or_else(|| anyhow::anyhow!("failed to determine data directory"))?;
    Ok(data_dir.join("mofa").join("skills").join("hub"))
}

pub fn default_skills_cache_root() -> Result<PathBuf> {
    let cache_dir = dirs_next::cache_dir()
        .ok_or_else(|| anyhow::anyhow!("failed to determine cache directory"))?;
    Ok(cache_dir.join("mofa").join("skills-hub"))
}

fn sanitize_bundle_path(path: &str) -> Result<PathBuf> {
    let candidate = Path::new(path);
    if candidate.is_absolute() {
        bail!("absolute bundle paths are not allowed: {}", path);
    }

    let mut clean = PathBuf::new();
    for component in candidate.components() {
        match component {
            Component::Normal(part) => clean.push(part),
            Component::CurDir => {}
            Component::ParentDir => bail!("path traversal is not allowed: {}", path),
            Component::RootDir | Component::Prefix(_) => {
                bail!("unsupported bundle path component: {}", path)
            }
        }
    }

    if clean.as_os_str().is_empty() {
        bail!("bundle path cannot be empty");
    }

    Ok(clean)
}

fn registry_dir(config: &SkillHubClientConfig) -> PathBuf {
    config.managed_root.join(".registry")
}

fn record_path(config: &SkillHubClientConfig, name: &str) -> PathBuf {
    registry_dir(config).join(format!("{}.json", sanitize_id(name)))
}

fn sanitize_id(value: &str) -> String {
    let sanitized: String = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' || ch == '.' {
                ch
            } else {
                '_'
            }
        })
        .collect();

    if sanitized.is_empty() {
        "_".to_string()
    } else {
        sanitized
    }
}

fn validate_skill_name(skill_name: &str) -> Result<&str> {
    let trimmed = skill_name.trim();
    if trimmed.is_empty() {
        bail!("invalid skill name: empty");
    }

    let mut components = Path::new(trimmed).components();
    let Some(Component::Normal(component)) = components.next() else {
        bail!(
            "invalid skill name '{}': must be a single path segment",
            skill_name
        );
    };
    if components.next().is_some() || component.to_str() != Some(trimmed) {
        bail!(
            "invalid skill name '{}': must be a single path segment",
            skill_name
        );
    }

    Ok(trimmed)
}

fn managed_skill_path(root: &Path, skill_name: &str) -> Result<PathBuf> {
    Ok(root.join(validate_skill_name(skill_name)?))
}

fn validate_bundle_files(bundle: &HubSkillBundle) -> Result<Vec<PathBuf>> {
    let mut wrote_skill_md = false;
    let mut files = Vec::with_capacity(bundle.files.len());

    for file in &bundle.files {
        let relative_path = sanitize_bundle_path(&file.path)?;
        if relative_path == Path::new("SKILL.md") {
            wrote_skill_md = true;
        }
        files.push(relative_path);
    }

    if !wrote_skill_md {
        bail!("hub bundle did not include SKILL.md");
    }

    Ok(files)
}

fn unique_work_dir(root: &Path, skill_name: &str, label: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    root.join(format!(".{}-{}-{}", label, sanitize_id(skill_name), nanos))
}

fn cache_key(catalog_url: &str) -> String {
    let mut hasher = DefaultHasher::new();
    catalog_url.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn read_bool_env(key: &str) -> Option<bool> {
    std::env::var(key)
        .ok()
        .and_then(|value| match value.to_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        })
}

fn now_epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Json, Router, routing::get};
    use serde_json::json;
    use tempfile::TempDir;
    use tokio::net::TcpListener;

    fn test_config(
        catalog_url: impl Into<String>,
        temp: &TempDir,
        auto_install_on_miss: bool,
    ) -> SkillHubClientConfig {
        SkillHubClientConfig::new(catalog_url)
            .with_managed_root(temp.path().join("managed"))
            .with_cache_root(temp.path().join("cache"))
            .with_auto_install_on_miss(auto_install_on_miss)
    }

    #[tokio::test]
    async fn test_search_catalog_filters_remote_results() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = Router::new().route(
            "/catalog",
            get(|| async {
                Json(json!({
                    "skills": [
                        {
                            "name": "pdf_processing",
                            "description": "Process PDF files",
                            "author": "OpenClaw",
                            "category": "documents",
                            "tags": ["pdf", "document"],
                            "latestVersion": "1.2.0",
                            "downloadUrl": "https://hub.local/pdf_processing-1.2.0.json"
                        },
                        {
                            "name": "web_scraping",
                            "description": "Scrape websites",
                            "author": "OpenClaw",
                            "category": "web",
                            "tags": ["web", "http"],
                            "latestVersion": "0.9.0",
                            "downloadUrl": "https://hub.local/web_scraping-0.9.0.json"
                        }
                    ]
                }))
            }),
        );
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let temp = TempDir::new().unwrap();
        let client = SkillHubClient::new(test_config(
            format!("http://{address}/catalog"),
            &temp,
            false,
        ))
        .unwrap();

        let results = client
            .search_with_category("pdf", Some("documents"))
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "pdf_processing");

        server.abort();
    }

    #[tokio::test]
    async fn test_sync_catalog_falls_back_to_cache_when_remote_is_unavailable() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let catalog_url = format!("http://{address}/catalog");
        let app = Router::new().route(
            "/catalog",
            get(|| async {
                Json(json!({
                    "skills": [
                        {
                            "name": "pdf_processing",
                            "description": "Process PDF files",
                            "compatibility": ["mofa"],
                            "latestVersion": "1.2.0",
                            "downloadUrl": "https://hub.local/pdf_processing-1.2.0.json"
                        }
                    ]
                }))
            }),
        );
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let temp = TempDir::new().unwrap();
        let online_client = SkillHubClient::new(test_config(&catalog_url, &temp, false)).unwrap();
        let online_entries = online_client.sync_catalog().await.unwrap();
        assert_eq!(online_entries.len(), 1);
        server.abort();

        let offline_client = SkillHubClient::new(test_config(&catalog_url, &temp, false)).unwrap();
        let cached_entries = offline_client.sync_catalog().await.unwrap();

        assert_eq!(cached_entries.len(), 1);
        assert_eq!(cached_entries[0].name, "pdf_processing");
    }

    #[tokio::test]
    async fn test_sync_catalog_does_not_use_cache_from_other_catalog_url() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let online_catalog_url = format!("http://{address}/catalog");
        let app = Router::new().route(
            "/catalog",
            get(|| async {
                Json(json!({
                    "skills": [
                        {
                            "name": "pdf_processing",
                            "description": "Process PDF files",
                            "compatibility": ["mofa"],
                            "latestVersion": "1.2.0",
                            "downloadUrl": "https://hub.local/pdf_processing-1.2.0.json"
                        }
                    ]
                }))
            }),
        );
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let temp = TempDir::new().unwrap();
        let online_client =
            SkillHubClient::new(test_config(&online_catalog_url, &temp, false)).unwrap();
        let online_entries = online_client.sync_catalog().await.unwrap();
        assert_eq!(online_entries.len(), 1);
        server.abort();

        let offline_client =
            SkillHubClient::new(test_config("http://127.0.0.1:9/catalog", &temp, false)).unwrap();
        let error = offline_client.sync_catalog().await.unwrap_err();
        assert!(error.to_string().contains("cache unavailable"));
    }

    #[tokio::test]
    async fn test_install_from_catalog_entry_downloads_bundle() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = Router::new().route(
            "/bundle",
            get(|| async {
                Json(json!({
                    "version": "1.2.0",
                    "files": [
                        {
                            "path": "SKILL.md",
                            "content": "---\nname: pdf_processing\ndescription: Process PDFs\n---\n# PDF"
                        },
                        {
                            "path": "scripts/extract.py",
                            "content": "print('downloaded')"
                        }
                    ]
                }))
            }),
        );
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let temp = TempDir::new().unwrap();
        let client =
            SkillHubClient::new(test_config("http://hub.local/catalog", &temp, false)).unwrap();
        let entry = HubSkillCatalogEntry {
            name: "pdf_processing".to_string(),
            description: "Process PDF files".to_string(),
            author: Some("OpenClaw".to_string()),
            tags: vec!["pdf".to_string()],
            category: Some("documents".to_string()),
            categories: Vec::new(),
            compatibility: vec!["mofa".to_string()],
            latest_version: Some("1.2.0".to_string()),
            download_url: Some(format!("http://{address}/bundle")),
            versions: Vec::new(),
            rating: Some(4.8),
            downloads: Some(42),
        };

        let installed = client
            .install_from_catalog_entry(temp.path(), &entry, None)
            .await
            .unwrap();

        assert_eq!(installed.version.as_deref(), Some("1.2.0"));
        assert!(installed.install_dir.join("SKILL.md").exists());
        assert!(
            installed
                .install_dir
                .join("scripts")
                .join("extract.py")
                .exists()
        );
        assert_eq!(
            installed.source_url.as_deref(),
            entry.download_url.as_deref()
        );

        server.abort();
    }

    #[tokio::test]
    async fn test_install_rejects_incompatible_skill() {
        let temp = TempDir::new().unwrap();
        let client =
            SkillHubClient::new(test_config("http://hub.local/catalog", &temp, false)).unwrap();
        let entry = HubSkillCatalogEntry {
            name: "other_skill".to_string(),
            description: "Other platform".to_string(),
            author: None,
            tags: vec![],
            category: None,
            categories: Vec::new(),
            compatibility: vec!["other-assistant".to_string()],
            latest_version: Some("1.0.0".to_string()),
            download_url: Some("http://127.0.0.1:9/bundle".to_string()),
            versions: Vec::new(),
            rating: None,
            downloads: None,
        };

        let error = client
            .install_from_catalog_entry(temp.path(), &entry, None)
            .await
            .unwrap_err();

        assert!(error.to_string().contains("not compatible"));
    }

    #[tokio::test]
    async fn test_install_rejects_compatibility_substring_false_positive() {
        let temp = TempDir::new().unwrap();
        let client =
            SkillHubClient::new(test_config("http://hub.local/catalog", &temp, false)).unwrap();
        let entry = HubSkillCatalogEntry {
            name: "other_skill".to_string(),
            description: "Other platform".to_string(),
            author: None,
            tags: vec![],
            category: None,
            categories: Vec::new(),
            compatibility: vec!["not-mofa".to_string()],
            latest_version: Some("1.0.0".to_string()),
            download_url: Some("http://127.0.0.1:9/bundle".to_string()),
            versions: Vec::new(),
            rating: None,
            downloads: None,
        };

        let error = client
            .install_from_catalog_entry(temp.path(), &entry, None)
            .await
            .unwrap_err();

        assert!(error.to_string().contains("not compatible"));
    }

    #[tokio::test]
    async fn test_install_accepts_compatibility_alias_labels() {
        let temp = TempDir::new().unwrap();
        let entry = HubSkillCatalogEntry {
            name: "portable_skill".to_string(),
            description: "Portable skill".to_string(),
            author: None,
            tags: vec![],
            category: None,
            categories: Vec::new(),
            compatibility: vec!["mofa-compatible".to_string(), "portable-skill".to_string()],
            latest_version: Some("1.0.0".to_string()),
            download_url: None,
            versions: Vec::new(),
            rating: None,
            downloads: None,
        };
        let install_root = temp.path().join("skills");
        std::fs::create_dir_all(&install_root).unwrap();
        let bundle = HubSkillBundle {
            version: Some("1.0.0".to_string()),
            files: vec![HubSkillBundleFile {
                path: "SKILL.md".to_string(),
                content: "---\nname: portable_skill\ndescription: Portable\n---\n# Portable"
                    .to_string(),
            }],
        };

        let installed = install_skill_bundle(&install_root, &entry.name, &bundle, None).unwrap();

        assert_eq!(installed.name, "portable_skill");
        assert!(installed.install_dir.join("SKILL.md").exists());
    }

    #[tokio::test]
    async fn test_install_and_update_registry_records() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let version = std::sync::Arc::new(tokio::sync::RwLock::new("1.0.0".to_string()));
        let server_version = version.clone();
        let bundle_version = version.clone();
        let app = Router::new()
            .route(
                "/catalog",
                get(move || {
                    let server_version = server_version.clone();
                    async move {
                        let current = server_version.read().await.clone();
                        Json(json!({
                            "skills": [
                                {
                                    "name": "pdf_processing",
                                    "description": "Process PDF files",
                                    "category": "documents",
                                    "compatibility": ["mofa"],
                                    "latestVersion": current,
                                    "downloadUrl": format!("http://{address}/bundle")
                                }
                            ]
                        }))
                    }
                }),
            )
            .route(
                "/bundle",
                get(move || {
                    let version = bundle_version.clone();
                    async move {
                        let current = version.read().await.clone();
                        Json(json!({
                            "version": current,
                            "files": [
                                {
                                    "path": "SKILL.md",
                                    "content": format!("---\nname: pdf_processing\ndescription: Process PDF files\nversion: \"{}\"\n---\n# PDF", current)
                                }
                            ]
                        }))
                    }
                }),
            );
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let temp = TempDir::new().unwrap();
        let client = SkillHubClient::new(test_config(
            format!("http://{address}/catalog"),
            &temp,
            true,
        ))
        .unwrap();

        let installed = client.install("pdf_processing", None).await.unwrap();
        assert_eq!(installed.version.as_deref(), Some("1.0.0"));

        *version.write().await = "1.1.0".to_string();
        let updated = client.update("pdf_processing").await.unwrap();
        assert_eq!(updated.version.as_deref(), Some("1.1.0"));

        let listed = client.list_installed().unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].version.as_deref(), Some("1.1.0"));
        let listed_value = serde_json::to_value(&listed[0]).unwrap();
        assert!(
            listed_value
                .get("updated_at_epoch_secs")
                .and_then(serde_json::Value::as_u64)
                .is_some()
        );

        server.abort();
    }

    #[test]
    fn test_install_bundle_writes_skill_files() {
        let temp = TempDir::new().unwrap();
        let bundle = HubSkillBundle {
            version: Some("1.2.0".to_string()),
            files: vec![
                HubSkillBundleFile {
                    path: "SKILL.md".to_string(),
                    content: "---\nname: pdf_processing\ndescription: Process PDFs\n---\n# PDF"
                        .to_string(),
                },
                HubSkillBundleFile {
                    path: "scripts/extract.py".to_string(),
                    content: "print('ok')".to_string(),
                },
            ],
        };

        let installed = install_skill_bundle(
            temp.path(),
            "pdf_processing",
            &bundle,
            Some("https://hub.local/pdf_processing-1.2.0.json"),
        )
        .unwrap();

        assert_eq!(installed.version.as_deref(), Some("1.2.0"));
        assert!(installed.install_dir.join("SKILL.md").exists());
        assert!(
            installed
                .install_dir
                .join("scripts")
                .join("extract.py")
                .exists()
        );
    }

    #[test]
    fn test_install_bundle_rejects_path_traversal() {
        let temp = TempDir::new().unwrap();
        let bundle = HubSkillBundle {
            version: Some("1.0.0".to_string()),
            files: vec![
                HubSkillBundleFile {
                    path: "SKILL.md".to_string(),
                    content: "# Unsafe".to_string(),
                },
                HubSkillBundleFile {
                    path: "../escape.sh".to_string(),
                    content: "echo nope".to_string(),
                },
            ],
        };

        let error = install_skill_bundle(temp.path(), "unsafe_skill", &bundle, None).unwrap_err();
        assert!(error.to_string().contains("path traversal"));
    }

    #[test]
    fn test_install_bundle_rejects_unsafe_skill_name() {
        let temp = TempDir::new().unwrap();
        let bundle = HubSkillBundle {
            version: Some("1.0.0".to_string()),
            files: vec![HubSkillBundleFile {
                path: "SKILL.md".to_string(),
                content: "# Safe".to_string(),
            }],
        };

        let error = install_skill_bundle(temp.path(), "../escape", &bundle, None).unwrap_err();
        assert!(error.to_string().contains("invalid skill name"));
    }

    #[test]
    fn test_install_bundle_preserves_existing_install_when_new_bundle_is_invalid() {
        let temp = TempDir::new().unwrap();
        let install_dir = temp.path().join("pdf_processing");
        fs::create_dir_all(&install_dir).unwrap();
        fs::write(install_dir.join("SKILL.md"), "# Old version").unwrap();

        let invalid_bundle = HubSkillBundle {
            version: Some("2.0.0".to_string()),
            files: vec![HubSkillBundleFile {
                path: "scripts/extract.py".to_string(),
                content: "print('broken')".to_string(),
            }],
        };

        let error =
            install_skill_bundle(temp.path(), "pdf_processing", &invalid_bundle, None).unwrap_err();
        assert!(error.to_string().contains("SKILL.md"));
        assert_eq!(
            fs::read_to_string(install_dir.join("SKILL.md")).unwrap(),
            "# Old version"
        );
    }

    #[test]
    fn test_remove_rejects_unsafe_skill_name() {
        let temp = TempDir::new().unwrap();
        let outside_dir = temp.path().join("outside_dir");
        fs::create_dir_all(&outside_dir).unwrap();

        let client =
            SkillHubClient::new(test_config("http://hub.local/catalog", &temp, false)).unwrap();
        let error = client.remove("../outside_dir").unwrap_err();

        assert!(error.to_string().contains("invalid skill name"));
        assert!(outside_dir.exists());
    }
}
