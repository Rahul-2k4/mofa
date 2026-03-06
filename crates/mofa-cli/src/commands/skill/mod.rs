use crate::context::CliContext;
use crate::output::Table;
use anyhow::bail;
use chrono::{DateTime, Utc};
use colored::Colorize;
use mofa_sdk::skills::{
    DEFAULT_SKILLS_HUB_CATALOG_URL, HubCacheStatus, HubSkillCatalogEntry, ManagedHubSkillRecord,
    SkillHubClient, SkillHubClientConfig,
};
use serde::Serialize;

const CONFIG_KEY_CATALOG_URL: &str = "skills.hub.catalog_url";
const CONFIG_KEY_AUTO_INSTALL: &str = "skills.hub.auto_install";
const CONFIG_KEY_COMPATIBILITY_TARGETS: &str = "skills.hub.compatibility_targets";

pub async fn run_search(
    ctx: &CliContext,
    catalog_url: Option<&str>,
    query: &str,
    category: Option<&str>,
) -> anyhow::Result<()> {
    println!("{} Searching skills hub", "→".green());
    if !query.is_empty() {
        println!("  Query: {}", query);
    }
    if let Some(category) = category {
        println!("  Category: {}", category);
    }
    println!();

    let client = hub_client(ctx, catalog_url)?;
    let results = client.search_with_category(query, category).await?;
    if results.is_empty() {
        println!("  No matching skills found.");
        return Ok(());
    }

    let rows: Vec<SearchRow> = results
        .into_iter()
        .map(|entry| {
            let category = entry
                .categories()
                .into_iter()
                .next()
                .unwrap_or_else(|| "-".to_string());
            SearchRow {
                name: entry.name,
                version: entry
                    .latest_version
                    .clone()
                    .unwrap_or_else(|| "-".to_string()),
                author: entry.author.clone().unwrap_or_else(|| "-".to_string()),
                category,
                description: entry.description,
            }
        })
        .collect();

    print_table(&rows)?;
    Ok(())
}

pub async fn run_categories(ctx: &CliContext, catalog_url: Option<&str>) -> anyhow::Result<()> {
    println!("{} Listing hub categories", "→".green());
    println!();

    let client = hub_client(ctx, catalog_url)?;
    let categories = client.categories().await?;
    if categories.is_empty() {
        println!("  No categories available.");
        return Ok(());
    }

    let rows: Vec<CategoryRow> = categories
        .into_iter()
        .map(|category| CategoryRow { category })
        .collect();
    print_table(&rows)?;
    Ok(())
}

pub async fn run_info(
    ctx: &CliContext,
    catalog_url: Option<&str>,
    skill_name: &str,
) -> anyhow::Result<()> {
    let client = hub_client(ctx, catalog_url)?;
    let entry = client
        .skill_details(skill_name)
        .await?
        .ok_or_else(|| anyhow::anyhow!("skill '{}' not found in remote catalog", skill_name))?;
    let installed = client.get_installed(skill_name)?;

    println!("{} {}", "Skill".green(), entry.name.bold());
    println!("  Description: {}", entry.description);
    println!(
        "  Author: {}",
        entry.author.clone().unwrap_or_else(|| "-".to_string())
    );
    println!(
        "  Latest version: {}",
        entry
            .latest_version
            .clone()
            .unwrap_or_else(|| "-".to_string())
    );
    println!(
        "  Categories: {}",
        join_or_dash(entry.categories().into_iter().collect())
    );
    println!(
        "  Compatibility: {}",
        join_or_dash(entry.compatibility.clone())
    );
    println!(
        "  Rating: {}",
        entry
            .rating
            .map(|value| format!("{value:.1}"))
            .unwrap_or_else(|| "-".to_string())
    );
    println!(
        "  Downloads: {}",
        entry
            .downloads
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".to_string())
    );
    println!(
        "  Installed: {}",
        installed
            .as_ref()
            .map(|record| format!("yes ({})", record.installed_path.display()))
            .unwrap_or_else(|| "no".to_string())
    );

    let versions: Vec<String> = entry
        .versions
        .into_iter()
        .map(|item| item.version)
        .collect();
    if !versions.is_empty() {
        println!("  Versions: {}", versions.join(", "));
    }

    Ok(())
}

pub async fn run_sync(ctx: &CliContext, catalog_url: Option<&str>) -> anyhow::Result<()> {
    let client = hub_client(ctx, catalog_url)?;
    let synced = client.sync_catalog().await?;
    println!(
        "{} Synced {} skills from hub catalog",
        "✓".green(),
        synced.len()
    );
    Ok(())
}

pub async fn run_install(
    ctx: &CliContext,
    catalog_url: Option<&str>,
    skill_ref: &str,
    version: Option<&str>,
) -> anyhow::Result<()> {
    let (skill_name, selected_version) = parse_skill_ref(skill_ref, version)?;
    let client = hub_client(ctx, catalog_url)?;
    let record = client
        .install(&skill_name, selected_version.as_deref())
        .await?;

    println!(
        "{} Installed skill '{}' from hub",
        "✓".green(),
        record.name.cyan()
    );
    if let Some(version) = record.version.as_deref() {
        println!("  Version: {}", version);
    }
    println!("  Path: {}", record.installed_path.display());
    Ok(())
}

pub fn run_list(ctx: &CliContext, catalog_url: Option<&str>) -> anyhow::Result<()> {
    println!("{} Listing installed hub skills", "→".green());
    println!();

    let client = hub_client(ctx, catalog_url)?;
    let installed = client.list_installed()?;
    if installed.is_empty() {
        println!("  No hub skills installed.");
        return Ok(());
    }

    let rows: Vec<InstalledRow> = installed
        .into_iter()
        .map(|entry| InstalledRow {
            name: entry.name,
            version: entry.version.unwrap_or_else(|| "-".to_string()),
            source: entry.source,
            installed_path: entry.installed_path.display().to_string(),
            installed_at: format_timestamp(entry.installed_at_epoch_secs),
            updated_at: entry
                .updated_at_epoch_secs
                .map(format_timestamp)
                .unwrap_or_else(|| "-".to_string()),
            last_used_at: entry
                .last_used_at_epoch_secs
                .map(format_timestamp)
                .unwrap_or_else(|| "-".to_string()),
        })
        .collect();

    print_table(&rows)?;
    Ok(())
}

pub async fn run_update(
    ctx: &CliContext,
    catalog_url: Option<&str>,
    skill_name: Option<&str>,
    all: bool,
) -> anyhow::Result<()> {
    let client = hub_client(ctx, catalog_url)?;

    if all {
        let updated = client.update_all().await?;
        println!("{} Updated {} skills", "✓".green(), updated.len());
        return Ok(());
    }

    let skill_name =
        skill_name.ok_or_else(|| anyhow::anyhow!("specify a skill name or pass --all"))?;
    let updated = client.update(skill_name).await?;
    println!(
        "{} Updated skill '{}' to {}",
        "✓".green(),
        updated.name.cyan(),
        updated.version.unwrap_or_else(|| "-".to_string())
    );
    Ok(())
}

pub fn run_remove(
    ctx: &CliContext,
    catalog_url: Option<&str>,
    skill_name: &str,
) -> anyhow::Result<()> {
    let client = hub_client(ctx, catalog_url)?;
    if client.remove(skill_name)? {
        println!("{} Removed skill '{}'", "✓".green(), skill_name.cyan());
    } else {
        println!("{} Skill '{}' was not installed", "→".green(), skill_name);
    }
    Ok(())
}

pub fn run_cache_status(ctx: &CliContext, catalog_url: Option<&str>) -> anyhow::Result<()> {
    let client = hub_client(ctx, catalog_url)?;
    let status = client.cache_status()?;

    println!("{} Hub cache status", "→".green());
    println!("  Path: {}", status.path.display());
    println!("  Exists: {}", status.exists);
    println!("  Entries: {}", status.entry_count);
    println!(
        "  Fetched at: {}",
        status
            .fetched_at_epoch_secs
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".to_string())
    );
    Ok(())
}

pub fn run_cache_clear(ctx: &CliContext, catalog_url: Option<&str>) -> anyhow::Result<()> {
    let client = hub_client(ctx, catalog_url)?;
    let cleared = client.clear_cache()?;
    if cleared {
        println!("{} Cleared hub cache", "✓".green());
    } else {
        println!("{} Hub cache was already empty", "→".green());
    }
    Ok(())
}

fn hub_client(ctx: &CliContext, catalog_url: Option<&str>) -> anyhow::Result<SkillHubClient> {
    let config = resolve_hub_config(ctx, catalog_url)?;
    SkillHubClient::new(config)
}

fn resolve_hub_config(
    ctx: &CliContext,
    catalog_url: Option<&str>,
) -> anyhow::Result<SkillHubClientConfig> {
    let persisted = mofa_foundation::config::load_global_config_from_dir(&ctx.config_dir)?;
    let catalog_url_from_config = persisted
        .get(CONFIG_KEY_CATALOG_URL)
        .cloned()
        .unwrap_or_else(|| DEFAULT_SKILLS_HUB_CATALOG_URL.to_string());
    let mut config = SkillHubClientConfig::new(catalog_url_from_config)
        .with_managed_root(ctx.data_dir.join("skills").join("hub"))
        .with_cache_root(ctx.cache_dir.join("skills-hub"));

    if let Some(value) = persisted
        .get(CONFIG_KEY_AUTO_INSTALL)
        .and_then(|value| parse_bool(value))
    {
        config.auto_install_on_miss = value;
    }
    if let Some(value) = persisted.get(CONFIG_KEY_COMPATIBILITY_TARGETS) {
        let targets = mofa_foundation::config::parse_global_string_list(value);
        if !targets.is_empty() {
            config.compatibility_targets = targets;
        }
    }
    if let Some(catalog_url) = catalog_url {
        config.catalog_url = catalog_url.to_string();
    }

    Ok(config)
}

pub fn parse_skill_ref(
    skill_ref: &str,
    version: Option<&str>,
) -> anyhow::Result<(String, Option<String>)> {
    if version.is_some()
        && skill_ref
            .rsplit_once('@')
            .is_some_and(|(name, parsed_version)| !name.is_empty() && !parsed_version.is_empty())
    {
        bail!("cannot use both inline skill version and --skill-version");
    }

    if let Some(version) = version {
        return Ok((skill_ref.to_string(), Some(version.to_string())));
    }

    if let Some((name, parsed_version)) = skill_ref.rsplit_once('@') {
        if !name.is_empty() && !parsed_version.is_empty() {
            return Ok((name.to_string(), Some(parsed_version.to_string())));
        }
    }

    Ok((skill_ref.to_string(), None))
}

fn parse_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn join_or_dash(values: Vec<String>) -> String {
    if values.is_empty() {
        "-".to_string()
    } else {
        values.join(", ")
    }
}

fn format_timestamp(epoch_secs: u64) -> String {
    DateTime::<Utc>::from_timestamp(epoch_secs as i64, 0)
        .map(|value| value.to_rfc3339())
        .unwrap_or_else(|| epoch_secs.to_string())
}

fn print_table<T: Serialize>(rows: &[T]) -> anyhow::Result<()> {
    let json = serde_json::to_value(rows)?;
    if let Some(arr) = json.as_array() {
        let table = Table::from_json_array(arr);
        println!("{}", table);
    }
    Ok(())
}

#[derive(Debug, Serialize)]
struct SearchRow {
    name: String,
    version: String,
    author: String,
    category: String,
    description: String,
}

#[derive(Debug, Serialize)]
struct CategoryRow {
    category: String,
}

#[derive(Debug, Serialize)]
struct InstalledRow {
    name: String,
    version: String,
    source: String,
    installed_path: String,
    installed_at: String,
    updated_at: String,
    last_used_at: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_parse_skill_ref_supports_inline_version() {
        let (name, version) = parse_skill_ref("pdf_processing@1.2.0", None).unwrap();
        assert_eq!(name, "pdf_processing");
        assert_eq!(version.as_deref(), Some("1.2.0"));
    }

    #[test]
    fn test_parse_skill_ref_rejects_conflicting_versions() {
        let error = parse_skill_ref("pdf_processing@1.2.0", Some("1.3.0")).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("cannot use both inline skill version and --skill-version")
        );
    }

    #[tokio::test]
    async fn test_global_config_overrides_hub_defaults() {
        let temp = TempDir::new().unwrap();
        let ctx = CliContext::with_temp_dir(temp.path()).await.unwrap();
        std::fs::write(
            ctx.config_dir.join("config.yml"),
            "skills.hub.catalog_url: https://hub.example/catalog\nskills.hub.auto_install: false\nskills.hub.compatibility_targets: mofa,portable\n",
        )
        .unwrap();

        let config = resolve_hub_config(&ctx, None).unwrap();
        assert_eq!(config.catalog_url, "https://hub.example/catalog");
        assert!(!config.auto_install_on_miss);
        assert_eq!(config.compatibility_targets, vec!["mofa", "portable"]);
    }

    #[tokio::test]
    async fn test_global_config_accepts_nested_yaml_skill_lists() {
        let temp = TempDir::new().unwrap();
        let ctx = CliContext::with_temp_dir(temp.path()).await.unwrap();
        std::fs::write(
            ctx.config_dir.join("config.yml"),
            r#"
skills:
  hub:
    catalog_url: https://hub.example/catalog
    auto_install: false
    compatibility_targets:
      - mofa
      - portable
"#,
        )
        .unwrap();

        let config = resolve_hub_config(&ctx, None).unwrap();
        assert_eq!(config.catalog_url, "https://hub.example/catalog");
        assert!(!config.auto_install_on_miss);
        assert_eq!(config.compatibility_targets, vec!["mofa", "portable"]);
    }
}
