#!/usr/bin/env python3
"""AgenticSeek CLI — inspired by crewAI's CLI pattern."""

from __future__ import annotations

import asyncio
import configparser
import os
import sys
from typing import Optional

import click

def _get_version() -> str:
    try:
        from importlib.metadata import version as _pkg_version
        return _pkg_version("agenticseek")
    except Exception:
        return "0.1.0"

__version__ = _get_version()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(config_path: str) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    read = cfg.read(config_path)
    if not read:
        click.secho(f"Warning: config file '{config_path}' not found, using defaults.", fg="yellow")
    return cfg


def _apply_overrides(
    cfg: configparser.ConfigParser,
    provider: Optional[str],
    model: Optional[str],
    headless: Optional[bool],
) -> None:
    """Apply CLI flag overrides on top of config values."""
    if provider is not None:
        cfg.set("MAIN", "provider_name", provider)
    if model is not None:
        cfg.set("MAIN", "provider_model", model)
    if headless is not None:
        cfg.set("BROWSER", "headless_browser", str(headless))


# ---------------------------------------------------------------------------
# Click group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(__version__, prog_name="agenticseek")
def agenticseek() -> None:
    """AgenticSeek — the open, local alternative to ManusAI."""


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

@agenticseek.command()
@click.option("--provider", default=None, help="LLM provider name (overrides config.ini).")
@click.option("--model", default=None, help="Model identifier (overrides config.ini).")
@click.option(
    "--headless/--no-headless",
    default=None,
    help="Run browser in headless mode (overrides config.ini).",
)
@click.option(
    "--config",
    "config_path",
    default="config.ini",
    show_default=True,
    help="Path to the config.ini file.",
)
def run(
    provider: Optional[str],
    model: Optional[str],
    headless: Optional[bool],
    config_path: str,
) -> None:
    """Start an interactive AgenticSeek session."""
    import warnings
    warnings.filterwarnings("ignore")

    from sources.llm_provider import Provider
    from sources.interaction import Interaction
    from sources.agents import (
        CoderAgent, CasualAgent, FileAgent,
        PlannerAgent, BrowserAgent,
    )
    from sources.browser import Browser, create_driver
    from sources.utility import pretty_print

    cfg = _load_config(config_path)
    _apply_overrides(cfg, provider, model, headless)

    stealth_mode = cfg.getboolean("BROWSER", "stealth_mode")
    personality_folder = "jarvis" if cfg.getboolean("MAIN", "jarvis_personality") else "base"
    languages = cfg["MAIN"]["languages"].split(" ")

    click.secho("Initializing AgenticSeek...", fg="cyan", bold=True)

    llm_provider = Provider(
        provider_name=cfg["MAIN"]["provider_name"],
        model=cfg["MAIN"]["provider_model"],
        server_address=cfg["MAIN"]["provider_server_address"],
        is_local=cfg.getboolean("MAIN", "is_local"),
    )

    browser = Browser(
        create_driver(
            headless=cfg.getboolean("BROWSER", "headless_browser"),
            stealth_mode=stealth_mode,
            lang=languages[0],
        ),
        anticaptcha_manual_install=stealth_mode,
    )

    agents = [
        CasualAgent(
            name=cfg["MAIN"]["agent_name"],
            prompt_path=f"prompts/{personality_folder}/casual_agent.txt",
            provider=llm_provider,
            verbose=False,
        ),
        CoderAgent(
            name="coder",
            prompt_path=f"prompts/{personality_folder}/coder_agent.txt",
            provider=llm_provider,
            verbose=False,
        ),
        FileAgent(
            name="File Agent",
            prompt_path=f"prompts/{personality_folder}/file_agent.txt",
            provider=llm_provider,
            verbose=False,
        ),
        BrowserAgent(
            name="Browser",
            prompt_path=f"prompts/{personality_folder}/browser_agent.txt",
            provider=llm_provider,
            verbose=False,
            browser=browser,
        ),
        PlannerAgent(
            name="Planner",
            prompt_path=f"prompts/{personality_folder}/planner_agent.txt",
            provider=llm_provider,
            verbose=False,
            browser=browser,
        ),
    ]

    interaction = Interaction(
        agents,
        tts_enabled=cfg.getboolean("MAIN", "speak"),
        stt_enabled=cfg.getboolean("MAIN", "listen"),
        recover_last_session=cfg.getboolean("MAIN", "recover_last_session"),
        langs=languages,
    )

    async def _loop() -> None:
        try:
            while interaction.is_active:
                interaction.get_user()
                if await interaction.think():
                    interaction.show_answer()
                    interaction.speak_answer()
        except Exception as exc:
            raise exc
        finally:
            if cfg.getboolean("MAIN", "save_session"):
                interaction.save_session()

    try:
        asyncio.run(_loop())
    except KeyboardInterrupt:
        click.secho("\nSession interrupted.", fg="yellow")


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------

@agenticseek.command(name="version")
@click.option(
    "--config",
    "config_path",
    default="config.ini",
    show_default=True,
    help="Path to the config.ini file.",
)
def version_cmd(config_path: str) -> None:
    """Show version and active configuration info."""
    click.echo(f"agenticseek version: {__version__}")
    cfg = _load_config(config_path)
    try:
        click.echo(f"provider:            {cfg['MAIN']['provider_name']}")
        click.echo(f"model:               {cfg['MAIN']['provider_model']}")
        click.echo(f"server address:      {cfg['MAIN']['provider_server_address']}")
    except (KeyError, configparser.NoSectionError):
        click.secho("Could not read provider details from config.", fg="yellow")


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

@agenticseek.command(name="config")
@click.option(
    "--config",
    "config_path",
    default="config.ini",
    show_default=True,
    help="Path to the config.ini file.",
)
def config_cmd(config_path: str) -> None:
    """Print the resolved configuration values (for debugging)."""
    cfg = _load_config(config_path)
    if not cfg.sections():
        click.secho("No configuration sections found.", fg="red")
        return
    for section in cfg.sections():
        click.secho(f"[{section}]", fg="cyan", bold=True)
        for key, value in cfg.items(section):
            click.echo(f"  {key} = {value}")
        click.echo()


# ---------------------------------------------------------------------------
# Backward-compatible entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    agenticseek()
