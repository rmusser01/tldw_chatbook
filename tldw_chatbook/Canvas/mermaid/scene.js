// Inert identity-free scenes; all coordinates pass here before serialization.
const sceneIntrinsics = Object.freeze({finite: Number.isFinite, stringify: JSON.stringify,
  create: Object.create, string: String, max: Math.max, min: Math.min, sign: Math.sign});
const typography = "font-family:monospace;font-size:16px;font-weight:normal;font-style:normal;line-height:24px";
let sceneBudget = null;
function sceneNode(tag, attributes = [], text = "", children = []) {
  sceneBudget.work();
  if (tag !== "div" && tag !== "p" && tag !== "pre") sceneBudget.charge("elements");
  // Charge each shallow record once, before attaching already-charged children.
  const ownBytes = utf8Bytes(sceneIntrinsics.stringify({tag, attributes, text, children: []})) + sceneIntrinsics.max(0, children.length-1);
  sceneBudget.charge("output", ownBytes);
  const node = sceneIntrinsics.create(null);
  node.tag = tag; node.attributes = attributes; node.text = text; node.children = children;
  return node;
}
function geometry(budget, ...values) {
  for (const value of values) {
    budget.work();
    if (!sceneIntrinsics.finite(value) || value < 0) budget.fail("geometry-limit");
  }
}
function sceneText(label, x, y, budget) {
  geometry(budget, x, y, label.width, label.height);
  extent(budget, x + label.width, y + label.height);
  return label.lines.map((line, i) => sceneNode("text", [["x", sceneIntrinsics.string(x)], ["y", sceneIntrinsics.string(y + 18 + i * 24)], ["style", typography]], line));
}
function sceneBox(box, shape, budget) {
  geometry(budget, box.x, box.y, box.width, box.height);
  const {x, y, width: w, height: h} = box;
  if (w <= 0 || h <= 0) budget.fail("geometry-limit");
  extent(budget, x+w, y+h);
  const path = shape === "diamond"
    ? `M${x + w/2} ${y}L${x + w} ${y + h/2}L${x + w/2} ${y + h}L${x} ${y + h/2}Z`
    : shape === "rounded"
      ? `M${x+8} ${y}H${x+w-8}Q${x+w} ${y} ${x+w} ${y+8}V${y+h-8}Q${x+w} ${y+h} ${x+w-8} ${y+h}H${x+8}Q${x} ${y+h} ${x} ${y+h-8}V${y+8}Q${x} ${y} ${x+8} ${y}Z`
      : `M${x} ${y}H${x+w}V${y+h}H${x}Z`;
  return sceneNode("path", [["d", path], ["fill", "white"], ["stroke", "black"]]);
}
function sceneArrow(points, dashed, budget) {
  for (const [x, y] of points) extent(budget, x, y);
  const [x, y] = points[points.length - 1], [px, py] = points[points.length - 2];
  const dx = sceneIntrinsics.sign(x-px), dy = sceneIntrinsics.sign(y-py);
  const a = [x-dx*8-dy*4, y-dy*8+dx*4], b = [x-dx*8+dy*4, y-dy*8-dx*4];
  geometry(budget, ...a, ...b);
  extent(budget, ...a); extent(budget, ...b);
  const attrs = [["d", points.map((p, i) => (i ? "L" : "M") + p.join(" ")).join("")], ["fill", "none"], ["stroke", "black"]];
  if (dashed) attrs.push(["stroke-dasharray", "6 4"]);
  return [sceneNode("path", attrs), sceneNode("path", [["d", `M${a.join(" ")}L${x} ${y}L${b.join(" ")}`], ["fill", "none"], ["stroke", "black"]])];
}
function extent(budget, x, y) {
  geometry(budget, x, y);
  if (x > 2048 || y > 4096) budget.fail("geometry-limit");
  budget.extentX = sceneIntrinsics.max(budget.extentX, x); budget.extentY = sceneIntrinsics.max(budget.extentY, y);
}
function finishScene(model, drawing, budget) {
  const {width, height, children} = drawing;
  geometry(budget, width, height);
  if (width <= 0 || height <= 0 || width > 2048 || height > 4096) budget.fail("geometry-limit");
  if (budget.extentX > width || budget.extentY > height) budget.fail("geometry-limit");
  budget.charge("area", width * height);
  const svg = sceneNode("svg", [["width", sceneIntrinsics.string(width)], ["height", sceneIntrinsics.string(height)], ["viewBox", `0 0 ${width} ${height}`], ["style", "display:block;max-width:none;flex-shrink:0"]], "", children);
  const root = sceneNode("div", [["style", "overflow:auto;max-width:100%"]], "", [
    sceneNode("p", [], model.kind === "flow" ? "Flowchart. Diagram source follows." : "Sequence diagram. Diagram source follows."), svg,
    sceneNode("pre", [], budget.source)]);
  const metrics = sceneIntrinsics.create(null);
  for (const [key, value] of budget.diagram) metrics[key] = value;
  const scene = {width, height, root, metrics};
  metrics.output = 0;
  let bytes = utf8Bytes(sceneIntrinsics.stringify(scene));
  // Including the byte count changes its decimal width; converge before charge.
  while (metrics.output !== bytes) { metrics.output = bytes; bytes = utf8Bytes(sceneIntrinsics.stringify(scene)); }
  budget.charge("output", bytes - budget.diagram.get("output"));
  return scene;
}
function layoutDiagram(model, budget) {
  sceneBudget = budget;
  try {
    return finishScene(model, model.kind === "flow" ? layoutFlow(model, budget) : layoutSequence(model, budget), budget);
  } finally { sceneBudget = null; }
}
