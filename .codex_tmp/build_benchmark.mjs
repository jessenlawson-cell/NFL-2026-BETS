import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const outputDir = "C:/Users/jesse/OneDrive/Documents/GitHub/NFL-2026-BETS/outputs/01a09b80-271b-7163-b535-2662a203fbb0";
const previewDir = "C:/Users/jesse/OneDrive/Documents/GitHub/NFL-2026-BETS/.codex_tmp/previews";
const outputPath = `${outputDir}/NFL_BETS_2026_Build_Benchmark.xlsx`;
const fontFamily = "Arial";
const colors = {
  navy: "#17365D",
  blue: "#2F75B5",
  paleBlue: "#D9EAF7",
  paleGreen: "#E2F0D9",
  green: "#375623",
  paleAmber: "#FFF2CC",
  amber: "#9C6500",
  paleRed: "#FCE4D6",
  red: "#9C0006",
  paleGray: "#F2F2F2",
  midGray: "#D9E1F2",
  text: "#222222",
  white: "#FFFFFF",
};

await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(previewDir, { recursive: true });

const workbook = Workbook.create();
const status = workbook.worksheets.add("Build Status");
const files = workbook.worksheets.add("Created Files");
const usage = workbook.worksheets.add("How to Use");

status.tabColor = colors.navy;
files.tabColor = colors.blue;
usage.tabColor = "#70AD47";

for (const sheet of [status, files, usage]) {
  sheet.showGridLines = false;
}

function title(sheet, text, widthRange) {
  sheet.getRange("A2").values = [[text]];
  sheet.getRange("A2").format.font = { name: fontFamily, size: 16, bold: true, color: colors.navy };
  sheet.getRange(widthRange).format.borders = {
    bottom: { style: "medium", color: colors.blue },
  };
}

function section(sheet, range, text) {
  sheet.mergeCells(range);
  const target = sheet.getRange(range);
  sheet.getRange(range.split(":")[0]).values = [[text]];
  target.format = {
    fill: colors.paleBlue,
    font: { name: fontFamily, size: 10, bold: true, color: colors.navy },
    borders: { preset: "outside", style: "thin", color: colors.blue },
    verticalAlignment: "center",
  };
}

function header(sheet, range) {
  const target = sheet.getRange(range);
  target.format = {
    fill: colors.navy,
    font: { name: fontFamily, size: 10, bold: true, color: colors.white },
    borders: {
      insideVertical: { style: "thin", color: colors.white },
      bottom: { style: "thin", color: colors.navy },
    },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
  };
}

title(status, "NFL 2026 Betting System: Build Benchmark", "A2:H2");
status.getRange("A3").values = [["Snapshot date"]];
status.getRange("B3").values = [[new Date("2026-09-13T00:00:00Z")]];
status.getRange("B3").setNumberFormat("yyyy-mm-dd");
status.getRange("A3:B3").format.font = { name: fontFamily, size: 10, italic: true, color: "#666666" };

section(status, "A5:H5", "Where the project stands now");
status.getRange("A6:C6").values = [["Item", "Current state", "What that means"]];
header(status, "A6:C6");
status.getRange("A7:C12").values = [
  ["Decision mode", "PASS-only", "The system cannot recommend a bet until a real model passes the locked 2025 test."],
  ["Docker image", "Built and verified", "The Python 3.13 container builds and runs the project validator."],
  ["Automated tests", 16, "All fixture tests pass inside the Python 3.13 Docker image."],
  ["Authoritative data", 0, "The eight CSV files have correct headers but no real NFL or market rows yet."],
  ["Real model", "Not trained", "Only fixture-based model tests have run. No production model artifact exists."],
  ["Live odds", "Not captured", "The odds command exists, but no API key or live request has been used."],
];
status.getRange("A7:C12").format.font = { name: fontFamily, size: 10, color: colors.text };
status.getRange("A7:C12").format.verticalAlignment = "center";
status.getRange("C7:C12").format.wrapText = true;
status.getRange("B9:B10").setNumberFormat("#,##0");
status.mergeCells("C6:E6");
status.getRange("C6").values = [["What that means"]];
status.getRange("C7:E12").merge(true);

section(status, "A14:H14", "Implementation benchmark");
status.getRange("A15:E15").values = [["Stage", "What is ready", "Status", "Evidence", "Next action"]];
header(status, "A15:E15");
const statusRows = [
  ["Repository foundation", "Folders, documentation, package layout, ignored runtime paths", "Verified", "Workspace audit", "Keep as the single source of truth"],
  ["Docker environment", "Pinned Python 3.13 image and mounted local folders", "Verified", "Image built; validator ran", "Use Docker commands from this workspace"],
  ["Database and schemas", "SQLite tables match all eight authoritative CSV contracts", "Verified", "Schema validator passed", "Populate through data sync"],
  ["Data synchronization", "nflreadpy loaders, cache, raw Parquet, curated CSVs, manifests", "Implemented", "Fixture integration test", "Run full 2009-2026 sync"],
  ["Odds snapshot", "One board request, atomic raw save, quota ledger, 60/40 consensus", "Implemented", "Fixture and failure tests", "Add API key, then run one smoke snapshot"],
  ["Feature build", "Past-only four-game EPA and success-rate EWMA", "Implemented", "Leakage mutation test", "Build after historical sync"],
  ["Model training", "Separate standardized ridge models for margin and total", "Implemented", "Fixture integration test", "Train after features validate"],
  ["Model testing", "One-time 2025 gate against no-vig closing market", "Implemented", "PASS-only gate tested", "Run once only after reviewing frozen metadata"],
  ["Historical NFL data", "Real schedules, plays, injuries and usage", "Pending", "Authoritative CSVs contain zero rows", "Run data sync"],
  ["Production features", "Historical pregame team snapshots", "Pending", "team_metrics.csv contains zero rows", "Run features build"],
  ["Production model", "Frozen version and promotion status", "Pending", "No model artifact exists", "Train and test a version"],
  ["Paper decision workflow", "Scan, settle, report and task installation", "Future", "Commands not implemented in this build", "Build after the production gate is understood"],
];
status.getRange(`A16:E${15 + statusRows.length}`).values = statusRows;
status.getRange(`A16:E${15 + statusRows.length}`).format.font = { name: fontFamily, size: 10, color: colors.text };
status.getRange(`A16:E${15 + statusRows.length}`).format.verticalAlignment = "top";
status.getRange(`B16:B${15 + statusRows.length}`).format.wrapText = true;
status.getRange(`D16:E${15 + statusRows.length}`).format.wrapText = true;
const statusRange = status.getRange(`C16:C${15 + statusRows.length}`);
statusRange.conditionalFormats.add("containsText", {
  text: "Verified",
  format: { fill: colors.paleGreen, font: { color: colors.green, bold: true } },
});
statusRange.conditionalFormats.add("containsText", {
  text: "Implemented",
  format: { fill: colors.paleBlue, font: { color: colors.navy, bold: true } },
});
statusRange.conditionalFormats.add("containsText", {
  text: "Pending",
  format: { fill: colors.paleAmber, font: { color: colors.amber, bold: true } },
});
statusRange.conditionalFormats.add("containsText", {
  text: "Future",
  format: { fill: colors.paleGray, font: { color: "#666666", bold: true } },
});

status.getRange("G6:H6").values = [["Status", "Count"]];
header(status, "G6:H6");
status.getRange("G7:G10").values = [["Verified"], ["Implemented"], ["Pending"], ["Future"]];
status.getRange("H7").formulas = [[`=COUNTIFS($C$16:$C$${15 + statusRows.length},G7)`]];
status.getRange("H7:H10").fillDown();
status.getRange("H7:H10").setNumberFormat("#,##0");
status.getRange("G7:H10").format.font = { name: fontFamily, size: 10, color: colors.text };
status.getRange("G12:H12").values = [["Authoritative files", 8]];
status.getRange("G13:H13").values = [["Paper betting status", "PASS-only"]];
status.getRange("G12:G13").format.font = { name: fontFamily, size: 10, bold: true, color: colors.navy };
status.getRange("A30").values = [["Source: local repository audit, Docker image build, container test suite and schema validation on 2026-09-13."]];
status.getRange("A30").format.font = { name: fontFamily, size: 9, italic: true, color: "#666666" };

status.getRange("A:A").format.columnWidth = 24;
status.getRange("B:B").format.columnWidth = 34;
status.getRange("C:C").format.columnWidth = 19;
status.getRange("D:D").format.columnWidth = 28;
status.getRange("E:E").format.columnWidth = 36;
status.getRange("F:F").format.columnWidth = 3;
status.getRange("G:G").format.columnWidth = 23;
status.getRange("H:H").format.columnWidth = 16;
status.getRange("6:6").format.rowHeight = 28;
status.getRange("15:15").format.rowHeight = 30;
status.getRange("16:27").format.rowHeight = 44;

title(files, "What Was Created and Why", "A2:E2");
files.getRange("A3").values = [["Each row points to a real part of the workspace. Paths are relative to the project folder."]];
files.getRange("A3").format.font = { name: fontFamily, size: 10, italic: true, color: "#666666" };
files.getRange("A5:E5").values = [["Area", "File or folder", "Why it exists", "How it is used", "Current state"]];
header(files, "A5:E5");
const fileRows = [
  ["Project guide", "README.md", "Plain-language entry point for setup and workflow", "Read first when opening the project", "Created and updated"],
  ["Project rules", "AGENTS.md", "Controls validation, leakage, betting and logging behavior", "Codex and developers follow it automatically", "Existing project authority"],
  ["Project status", "PROJECT_STATE.md", "Records season, model and betting status", "Update only when a real stage changes", "PASS-only"],
  ["Data definitions", "DATA_DICTIONARY.md", "Explains every authoritative dataset", "Use when checking columns and keys", "Created and updated"],
  ["Model rules", "MODEL_SPEC.md", "Freezes targets, chronology, calibration and promotion rules", "Review before training or testing", "Created"],
  ["Container", "Dockerfile", "Builds the exact Python 3.13 environment", "Docker uses it to create the local image", "Built successfully"],
  ["Container runner", "docker-compose.yml", "Connects the image to local data, models, reports and logs", "Runs each nfl-bets command", "Validated"],
  ["Dependencies", "requirements.txt", "Lists direct Python packages", "Used by developers to understand the stack", "Pinned"],
  ["Dependency lock", "requirements.lock", "Pins every resolved package version", "Docker installs this exact set", "Verified in Python 3.13"],
  ["Local settings", ".env.example", "Shows safe local environment variables", "Copy to .env and add the odds key", ".env remains untracked"],
  ["Package entry", "pyproject.toml", "Defines the installable package and nfl-bets command", "Python installs the CLI from here", "Created"],
  ["CLI", "src/nfl_bets/cli.py", "Provides the human-facing commands", "Routes each command to the correct module", "Implemented"],
  ["Paths", "src/nfl_bets/config.py", "Keeps all local paths and settings consistent", "Maps Windows folders into container folders", "Implemented"],
  ["Schemas", "src/nfl_bets/schemas.py", "Defines authoritative CSV columns and keys", "Keeps CSV and SQLite contracts aligned", "Implemented"],
  ["Database", "src/nfl_bets/db.py", "Creates SQLite tables and append-only registries", "Stores runs, raw snapshots, odds and model history", "Implemented"],
  ["Database setup", "scripts/init_db.py", "Provides a direct initialization script", "Creates the local database outside the CLI if needed", "Implemented"],
  ["NFL data", "src/nfl_bets/data/sync.py", "Downloads official nflverse data through nflreadpy", "Writes cached Parquet, manifests, CSVs and SQLite rows", "Implemented, not fully run"],
  ["Odds request", "src/nfl_bets/odds/client.py", "Makes one quota-controlled NFL board request", "Saves raw bytes before parsing and records quota use", "Implemented, no live call"],
  ["Odds consensus", "src/nfl_bets/odds/consensus.py", "Normalizes prices and builds Pinnacle-anchored consensus", "Applies 60/40 logic and rejects zero-juice flatlines", "Implemented"],
  ["Features", "src/nfl_bets/features/build.py", "Builds prior-game EPA and success-rate signals", "Applies shift(1), four-game EWMA and early-season priors", "Implemented"],
  ["Residual mapping", "src/nfl_bets/model/residuals.py", "Turns projections into win, push and loss probabilities", "Preserves mass at spread key numbers", "Implemented"],
  ["Model pipeline", "src/nfl_bets/model/training.py", "Trains margin and total ridge models and tests promotion", "Locks development through 2024 and consumes 2025 once", "Implemented"],
  ["Validation", "src/nfl_bets/validation.py", "Checks schemas, duplicates, joins and metadata", "Run before model or market work", "Implemented"],
  ["Games", "games.csv", "Canonical schedule, results and closing market", "Populated by data sync", "Header only"],
  ["Team features", "team_metrics.csv", "One past-only team snapshot per game", "Populated by features build", "Header only"],
  ["Odds history", "market_odds.csv", "Append-only normalized bookmaker quotes", "Populated by odds snapshot", "Header only"],
  ["Injuries", "injuries.csv", "Validated injury-report input", "Populated by data sync; disabled as V1 feature", "Header only"],
  ["Player usage", "player_usage.csv", "Validated snap-count input", "Populated by data sync; disabled as V1 feature", "Header only"],
  ["Coverage", "coverage_metrics.csv", "Reserved schema for a future validated source", "Never synthesize values", "Unavailable by design"],
  ["Model history", "model_history.csv", "Permanent train and test registry", "Keeps failed and PASS-only versions", "Header only"],
  ["Bet log", "bet_log.csv", "Permanent paper-decision and settlement record", "Will retain PASS decisions and losses", "Header only"],
  ["Raw and cache", "data/raw and data/cache", "Keeps provider payloads and reusable downloads outside Git", "Mounted into Docker and retained locally", "Folders created"],
  ["Runtime", "data/runtime", "Holds the local SQLite database", "Mounted into Docker and retained locally", "Database initialized"],
  ["Models", "artifacts/models", "Holds frozen model files outside Git", "Written by model train", "Empty"],
  ["Reports", "reports", "Destination for model and test reports", "Written by model test and later report commands", "Folder created"],
  ["Tests", "tests", "Protects odds math, leakage, schemas and one-time test behavior", "Run with pytest locally or in Docker", "16 passing"],
  ["Synthetic fixture", "tests/fixtures/randomized_team_metrics.csv", "Quarantines the old randomized data", "Used only for deterministic tests", "Not betting evidence"],
];
files.getRange(`A6:E${5 + fileRows.length}`).values = fileRows;
files.getRange(`A6:E${5 + fileRows.length}`).format.font = { name: fontFamily, size: 10, color: colors.text };
files.getRange(`A6:E${5 + fileRows.length}`).format.verticalAlignment = "top";
files.getRange(`C6:E${5 + fileRows.length}`).format.wrapText = true;
files.getRange("A:A").format.columnWidth = 21;
files.getRange("B:B").format.columnWidth = 43;
files.getRange("C:C").format.columnWidth = 38;
files.getRange("D:D").format.columnWidth = 42;
files.getRange("E:E").format.columnWidth = 24;
files.getRange("5:5").format.rowHeight = 30;
files.getRange(`6:${5 + fileRows.length}`).format.rowHeight = 43;
files.freezePanes.freezeRows(5);

title(usage, "How to Access and Use the Project", "A2:D2");
section(usage, "A4:D4", "Where to find it");
usage.getRange("A5:C5").values = [["Access point", "Location", "What you will see"]];
header(usage, "A5:C5");
usage.getRange("A6:C9").values = [
  ["Windows File Explorer", "C:\\Users\\jesse\\OneDrive\\Documents\\GitHub\\NFL-2026-BETS", "Every project file plus retained data and outputs"],
  ["VS Code", "NFL-2026-BETS.code-workspace", "The project code, tests and terminal"],
  ["Docker Desktop", "Images > nfl-2026-bets:local", "The verified Python 3.13 image"],
  ["Codex", "This NFL-2026-BETS workspace", "The same files with assisted editing and review"],
];
usage.getRange("A6:C9").format.font = { name: fontFamily, size: 10, color: colors.text };
usage.getRange("B6:C9").format.wrapText = true;

section(usage, "A11:D11", "Make Docker available in this PowerShell window");
usage.getRange("A12").values = [["Run these two lines once per new terminal if docker is not recognized:"]];
usage.getRange("A13").values = [["$dockerBin='C:\\Users\\jesse\\AppData\\Local\\Programs\\DockerDesktop\\resources\\bin'"]];
usage.getRange("A14").values = [["$env:Path=\"$dockerBin;$env:Path\""]];
usage.getRange("A13:A14").format = {
  fill: "#F7F7F7",
  font: { name: "Consolas", size: 9, color: colors.text },
  borders: { preset: "outside", style: "thin", color: "#D9D9D9" },
};

section(usage, "A16:D16", "Recommended order");
usage.getRange("A17:D17").values = [["Step", "Command", "What happens", "Important note"]];
header(usage, "A17:D17");
const commandRows = [
  [1, "docker compose build", "Builds or refreshes the local Python 3.13 image", "Already completed successfully for this benchmark"],
  [2, "docker compose run --rm nfl-bets db init", "Creates the local SQLite tables", "Data persists in data/runtime"],
  [3, "docker compose run --rm nfl-bets data sync --through 2026", "Downloads 2009-2026 nflverse history and current files", "This is the first large real-data operation"],
  [4, "docker compose run --rm nfl-bets features build --as-of <ISO timestamp>", "Builds past-only team snapshots", "Timestamp must include a timezone"],
  [5, "docker compose run --rm nfl-bets validate", "Checks schemas, duplicates, joins and metadata", "Stop if validation fails"],
  [6, "docker compose run --rm nfl-bets model train --version 1.0.0", "Runs walk-forward development through 2024 and freezes the candidate", "Review metadata before testing"],
  [7, "docker compose run --rm nfl-bets model test --version 1.0.0 --season 2025", "Runs the untouched test and records PROMOTED or PASS_ONLY", "May run only once for that version"],
  [8, "docker compose run --rm nfl-bets odds snapshot --slot manual", "Captures one live spreads and totals board", "Requires the local odds API key"],
];
usage.getRange(`A18:D${17 + commandRows.length}`).values = commandRows;
usage.getRange(`A18:D${17 + commandRows.length}`).format.font = { name: fontFamily, size: 10, color: colors.text };
usage.getRange(`A18:D${17 + commandRows.length}`).format.verticalAlignment = "top";
usage.getRange(`B18:D${17 + commandRows.length}`).format.wrapText = true;
usage.getRange("A18:A25").setNumberFormat("0");

section(usage, "A27:D27", "Odds API key and retained files");
usage.getRange("A28:C31").values = [
  ["API key", "Copy .env.example to .env and enter THE_ODDS_API_KEY in the local copy.", ".env is ignored by Git and is not placed in commands or logs."],
  ["Docker runs", "The --rm option removes the temporary container after each command.", "This does not remove the mounted project data."],
  ["Persistent data", "data, artifacts, reports and logs are mounted Windows folders.", "You can inspect them directly in File Explorer."],
  ["Current safety state", "The project stays PASS-only until a candidate passes its locked gate.", "No live or historical result currently supports a wager."],
];
usage.getRange("A28:C31").format.font = { name: fontFamily, size: 10, color: colors.text };
usage.getRange("B28:C31").format.wrapText = true;
usage.getRange("A:A").format.columnWidth = 18;
usage.getRange("B:B").format.columnWidth = 68;
usage.getRange("C:C").format.columnWidth = 46;
usage.getRange("D:D").format.columnWidth = 43;
usage.getRange("5:5").format.rowHeight = 28;
usage.getRange("17:17").format.rowHeight = 30;
usage.getRange("18:25").format.rowHeight = 48;
usage.getRange("28:31").format.rowHeight = 43;

workbook.recalculate();

const statusCheck = await workbook.inspect({
  kind: "table",
  range: "Build Status!A1:H30",
  include: "values,formulas",
  tableMaxRows: 30,
  tableMaxCols: 8,
});
console.log(statusCheck.ndjson);

const filesCheck = await workbook.inspect({
  kind: "table",
  range: `Created Files!A1:E${5 + fileRows.length}`,
  include: "values,formulas",
  tableMaxRows: 45,
  tableMaxCols: 5,
});
console.log(filesCheck.ndjson);

const usageCheck = await workbook.inspect({
  kind: "table",
  range: "How to Use!A1:D31",
  include: "values,formulas",
  tableMaxRows: 35,
  tableMaxCols: 4,
});
console.log(usageCheck.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

for (const [sheetName, fileName] of [
  ["Build Status", "build-status.png"],
  ["Created Files", "created-files.png"],
  ["How to Use", "how-to-use.png"],
]) {
  const preview = await workbook.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
  await fs.writeFile(`${previewDir}/${fileName}`, new Uint8Array(await preview.arrayBuffer()));
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(JSON.stringify({ outputPath, previewDir }));
