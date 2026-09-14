import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const path = "C:/Users/jesse/OneDrive/Documents/GitHub/NFL-2026-BETS/outputs/01a09b80-271b-7163-b535-2662a203fbb0/NFL_BETS_2026_Build_Benchmark.xlsx";
const input = await FileBlob.load(path);
const workbook = await SpreadsheetFile.importXlsx(input);

console.log(JSON.stringify(workbook.inspect({
  kind: "table",
  range: "'Build Status'!G6:H13",
  include: "values,formulas",
  tableMaxRows: 20,
  tableMaxCols: 4,
})));
console.log(JSON.stringify(workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 50 },
  summary: "final saved workbook formula error scan",
})));
