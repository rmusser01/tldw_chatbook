// Unicode 16.0 / UAX29 revision45. Inputs and table generation are pinned.
function hasProperty(cp, name) {
  const spans = unicodeTables[name] || [];
  let lo = 0, hi = spans.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >>> 1;
    if (cp < spans[mid][0]) hi = mid - 1;
    else if (cp > spans[mid][1]) lo = mid + 1;
    else return true;
  }
  return false;
}
function graphemeProperty(cp) {
  for (const prop of ["CR", "LF", "Control", "Extend", "ZWJ", "Regional_Indicator", "Prepend", "SpacingMark", "L", "V", "T", "LV", "LVT"]) {
    if (hasProperty(cp, "GCB_" + prop)) return prop;
  }
  return "Other";
}
function segmentGraphemes(text) {
  const chars = Array.from(text);
  const points = chars.map(c => c.codePointAt(0));
  const props = points.map(graphemeProperty);
  const out = [];
  let current = "";
  for (let i = 0; i < chars.length; i += 1) {
    let joined = false;
    if (i > 0) {
      const a = props[i - 1], b = props[i];
      if (a === "CR" && b === "LF") joined = true; // GB3
      else if (["Control", "CR", "LF"].includes(a) || ["Control", "CR", "LF"].includes(b)) joined = false; // GB4/5
      else if (a === "L" && ["L", "V", "LV", "LVT"].includes(b)) joined = true;
      else if (["LV", "V"].includes(a) && ["V", "T"].includes(b)) joined = true;
      else if (["LVT", "T"].includes(a) && b === "T") joined = true;
      else if (["Extend", "ZWJ", "SpacingMark"].includes(b) || a === "Prepend") joined = true;
      else {
        if (hasProperty(points[i], "InCB_Consonant")) {
          let j = i - 1, linker = false;
          while (j >= 0 && (hasProperty(points[j], "InCB_Extend") || hasProperty(points[j], "InCB_Linker"))) {
            linker ||= hasProperty(points[j], "InCB_Linker"); j -= 1;
          }
          joined = linker && j >= 0 && hasProperty(points[j], "InCB_Consonant"); // GB9c
        }
        if (!joined && hasProperty(points[i], "Extended_Pictographic") && a === "ZWJ") {
          let j = i - 2;
          while (j >= 0 && props[j] === "Extend") j -= 1;
          joined = j >= 0 && hasProperty(points[j], "Extended_Pictographic"); // GB11
        }
        if (!joined && a === "Regional_Indicator" && b === "Regional_Indicator") {
          let count = 0;
          for (let j = i - 1; j >= 0 && props[j] === "Regional_Indicator"; j -= 1) count += 1;
          joined = count % 2 === 1; // GB12/13
        }
      }
    }
    if (!joined && current) { out.push(current); current = ""; }
    current += chars[i];
  }
  if (current) out.push(current);
  return out;
}
function graphemeWidth(cluster) {
  let cells = 0;
  const cps = Array.from(cluster, c => c.codePointAt(0));
  for (const cp of cps) {
    const prop = graphemeProperty(cp);
    if (["Extend", "ZWJ", "Control", "CR", "LF"].includes(prop)) continue;
    cells = Math.max(cells, hasProperty(cp, "Wide") || hasProperty(cp, "Emoji_Presentation") ? 2 : 1);
  }
  if ((cells > 0 && cps.includes(0xfe0f)) || (cps.includes(0x200d) && cps.some(cp => hasProperty(cp, "Extended_Pictographic")))) cells = 2;
  return cells * 8;
}
