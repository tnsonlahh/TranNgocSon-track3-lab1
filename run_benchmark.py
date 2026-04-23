from __future__ import annotations
import json
import os
from pathlib import Path
import typer
from rich import print
from dotenv import load_dotenv
from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.mock_runtime import MockRuntime
from src.reflexion_lab.openai_runtime import OpenAIRuntime
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.utils import load_dataset, save_jsonl
app = typer.Typer(add_completion=False)

@app.command()
def main(
    dataset: str = "data/hotpot_mini.json",
    out_dir: str = "outputs/sample_run",
    reflexion_attempts: int = 3,
    mode: str = "mock",
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    openai_api_key_env: str = "OPENAI_API_KEY",
) -> None:
    load_dotenv()
    examples = load_dataset(dataset)

    if mode == "mock":
        runtime = MockRuntime()
    elif mode == "openai":
        api_key = os.getenv(openai_api_key_env)
        if not api_key:
            raise typer.BadParameter(f"Missing API key in env var: {openai_api_key_env}")
        runtime = OpenAIRuntime(model=model, api_key=api_key, temperature=temperature)
    else:
        raise typer.BadParameter("mode must be either 'mock' or 'openai'")

    react = ReActAgent(runtime=runtime)
    reflexion = ReflexionAgent(max_attempts=reflexion_attempts, runtime=runtime)
    react_records = [react.run(example) for example in examples]
    reflexion_records = [reflexion.run(example) for example in examples]
    all_records = react_records + reflexion_records
    out_path = Path(out_dir)
    save_jsonl(out_path / "react_runs.jsonl", react_records)
    save_jsonl(out_path / "reflexion_runs.jsonl", reflexion_records)
    report = build_report(all_records, dataset_name=Path(dataset).name, mode=mode)
    if mode == "openai":
        report.meta["model"] = model
        report.meta["temperature"] = temperature
    json_path, md_path = save_report(report, out_path)
    print(f"[green]Saved[/green] {json_path}")
    print(f"[green]Saved[/green] {md_path}")
    print(json.dumps(report.summary, indent=2))

if __name__ == "__main__":
    app()
