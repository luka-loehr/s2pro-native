//! Credentials and endpoint configuration — all resolved at runtime, nothing
//! baked into the binary.
//!
//! This repository is public: there is no project id, key, or token anywhere in
//! the source. The access token comes from `gcloud` (or an env override, for CI
//! and workload-identity setups) and the project from the active gcloud config.

use anyhow::{Context, Result, bail};
use std::process::Command;

/// Default TTS model. `global` is the only location that serves the preview
/// TTS models today.
const DEFAULT_TTS_MODEL: &str = "gemini-3.1-flash-tts-preview";
const DEFAULT_TTS_LOCATION: &str = "global";
/// Default text model, for passage authoring and `--verify` transcription.
const DEFAULT_TEXT_MODEL: &str = "gemini-3.6-flash";
const DEFAULT_TEXT_LOCATION: &str = "europe-west4";

pub struct Config {
    pub project: String,
    pub token: String,
    pub tts_model: String,
    pub tts_location: String,
    pub text_model: String,
    pub text_location: String,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            project: project()?,
            token: token()?,
            tts_model: env_or("S2P_TTS_MODEL", DEFAULT_TTS_MODEL),
            tts_location: env_or("S2P_TTS_LOCATION", DEFAULT_TTS_LOCATION),
            text_model: env_or("S2P_TEXT_MODEL", DEFAULT_TEXT_MODEL),
            text_location: env_or("S2P_TEXT_LOCATION", DEFAULT_TEXT_LOCATION),
        })
    }
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| default.to_owned())
}

/// Project id: `GOOGLE_CLOUD_PROJECT`, else the active gcloud config.
fn project() -> Result<String> {
    if let Ok(project) = std::env::var("GOOGLE_CLOUD_PROJECT") {
        let project = project.trim().to_owned();
        if !project.is_empty() {
            return Ok(project);
        }
    }
    let project = gcloud(&["config", "get-value", "project"]).context(
        "could not determine the GCP project — set GOOGLE_CLOUD_PROJECT or run \
         `gcloud config set project <id>`",
    )?;
    if project.is_empty() || project == "(unset)" {
        bail!(
            "no active gcloud project — set GOOGLE_CLOUD_PROJECT or run \
             `gcloud config set project <id>`"
        );
    }
    Ok(project)
}

/// Access token: `S2P_GEMINI_ACCESS_TOKEN`, else `gcloud auth print-access-token`.
///
/// Shelling out to gcloud is deliberate: it means this tool never reads, parses,
/// or stores a credential itself. Whatever gcloud is logged in as is what gets
/// used — user ADC on a laptop, a service account in CI.
fn token() -> Result<String> {
    if let Ok(token) = std::env::var("S2P_GEMINI_ACCESS_TOKEN") {
        let token = token.trim().to_owned();
        if !token.is_empty() {
            return Ok(token);
        }
    }
    let token = gcloud(&["auth", "print-access-token"]).context(
        "could not get an access token — run `gcloud auth login` (or set \
         S2P_GEMINI_ACCESS_TOKEN)",
    )?;
    if token.is_empty() {
        bail!("`gcloud auth print-access-token` returned nothing — run `gcloud auth login`");
    }
    Ok(token)
}

fn gcloud(args: &[&str]) -> Result<String> {
    let out = Command::new("gcloud")
        .args(args)
        .output()
        .with_context(|| format!("running `gcloud {}`", args.join(" ")))?;
    if !out.status.success() {
        bail!(
            "`gcloud {}` failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_owned())
}
