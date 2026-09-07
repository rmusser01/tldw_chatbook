// The returned value is retained by the worker; nothing is installed globally.
function renderDiagrams(records) {
  const budget = new DiagramBudget(), scenes = [];
  for (const record of records) {
    const model = parseMermaid(record.source, budget);
    const scene = layoutDiagram(model, budget);
    scenes.push(scene);
  }
  return scenes;
}
return Object.freeze({parseMermaid, DiagramBudget, DiagramError, segmentGraphemes, graphemeWidth, layoutDiagram, renderDiagrams});
