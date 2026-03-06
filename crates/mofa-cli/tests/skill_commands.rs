#![cfg(test)]

use assert_cmd::Command;
use axum::{Json, Router, routing::get};
use predicates::prelude::*;
use serde_json::json;
use tempfile::TempDir;
use tokio::net::TcpListener;

fn write_nested_global_config(temp: &TempDir, catalog_url: &str) {
    let config_dir = temp.path().join("xdg-config").join("mofa");
    std::fs::create_dir_all(&config_dir).unwrap();
    std::fs::write(
        config_dir.join("config.yml"),
        format!(
            r#"
skills:
  hub:
    catalog_url: {catalog_url}
    auto_install: true
    compatibility_targets:
      - mofa
      - portable
"#
        ),
    )
    .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_skill_install_uses_nested_yaml_global_hub_config() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let app = Router::new()
        .route(
            "/catalog",
            get(move || async move {
                Json(json!({
                    "skills": [
                        {
                            "name": "portable_skill",
                            "description": "Portable skill",
                            "author": "OpenClaw",
                            "latestVersion": "1.2.0",
                            "downloadUrl": format!("http://{address}/bundle"),
                            "compatibility": ["portable-skill"]
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
                            "content": "---\nname: portable_skill\ndescription: Portable skill\n---\n# Portable\nUse this skill"
                        }
                    ]
                }))
            }),
        );
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let temp = TempDir::new().unwrap();
    std::fs::create_dir_all(temp.path().join("xdg-data")).unwrap();
    std::fs::create_dir_all(temp.path().join("xdg-cache")).unwrap();
    write_nested_global_config(&temp, &format!("http://{address}/catalog"));

    let mut cmd = Command::cargo_bin("mofa").unwrap();
    cmd.arg("skill")
        .arg("install")
        .arg("portable_skill")
        .env("XDG_CONFIG_HOME", temp.path().join("xdg-config"))
        .env("XDG_DATA_HOME", temp.path().join("xdg-data"))
        .env("XDG_CACHE_HOME", temp.path().join("xdg-cache"))
        .env("NO_COLOR", "1");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Installed skill"))
        .stdout(predicate::str::contains("portable_skill"));

    assert!(
        temp.path()
            .join("xdg-data")
            .join("mofa")
            .join("skills")
            .join("hub")
            .join("portable_skill")
            .join("SKILL.md")
            .exists()
    );

    server.abort();
}
