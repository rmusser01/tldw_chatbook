function layoutFlow(model, budget) {
  const nodes = new Map(), indegree = new Map(), ranks = new Map(), outgoing = new Map(), sourceIndex = new Map();
  for (const node of model.nodes) {
    budget.work(); sourceIndex.set(node.id, nodes.size); nodes.set(node.id, node); indegree.set(node.id, 0); ranks.set(node.id, 0); outgoing.set(node.id, []);
  }
  for (const edge of model.edges) {
    budget.work(); indegree.set(edge.to, indegree.get(edge.to)+1); outgoing.get(edge.from).push(edge);
  }
  const order = [];
  while (order.length < nodes.size) {
    let next = null;
    for (const node of model.nodes) {
      budget.work(); if (indegree.get(node.id) === 0) { next = node.id; break; }
    }
    if (next === null) budget.fail("cycle");
    indegree.set(next, -1); order.push(next);
    for (const edge of outgoing.get(next)) {
      budget.work(); ranks.set(edge.to, Math.max(ranks.get(edge.to), ranks.get(next)+1)); indegree.set(edge.to, indegree.get(edge.to)-1);
    }
  }
  // No coordinate/label state is allocated until acyclicity is established.
  const rows = [], boxes = new Map(), edgeLabels = model.edges.map(e => wrapLabel(e.label, budget, 128));
  for (const id of order) { budget.work(); const rank = ranks.get(id); (rows[rank] ||= []).push(id); }
  const positions = new Map();
  for (const row of rows) for (let i = 0; i < row.length; i += 1) { budget.work(); positions.set(row[i], i); }
  for (let sweep = 0; sweep < 2; sweep += 1) {
    for (const forward of [true, false]) {
      const indices = rows.map((_r, i) => i); if (!forward) indices.reverse();
      for (const index of indices) {
        const scores = new Map();
        for (const id of rows[index]) {
          let total = 0, count = 0;
          for (const edge of model.edges) {
            budget.work(); const other = forward && edge.to === id ? edge.from : !forward && edge.from === id ? edge.to : null;
            if (other !== null) { total += positions.get(other); count += 1; }
          }
          scores.set(id, count ? total/count : positions.get(id));
        }
        rows[index].sort((a,b) => { budget.work(); return scores.get(a)-scores.get(b) || sourceIndex.get(a)-sourceIndex.get(b); });
        for (let i = 0; i < rows[index].length; i += 1) { budget.work(); positions.set(rows[index][i], i); }
      }
    }
  }
  const lr = model.direction === "LR";
  let major = 24, crossMax = 0;
  const lanes = new Map();
  for (let rank = 0; rank < rows.length; rank += 1) {
    const row = rows[rank];
    let cross = 24, thick = 0;
    for (const id of row) {
      budget.work(); const node = nodes.get(id), label = wrapLabel(node.label, budget);
      const factor = node.shape === "diamond" ? 2 : 1;
      const width = Math.max(64, (label.width+32)*factor), height = (label.height+24)*factor;
      boxes.set(id, {x: lr ? major : cross, y: lr ? cross : major, width, height, label});
      let crossSpace = lr ? height : width;
      for (let i = 0; i < model.edges.length; i += 1) {
        budget.work();
        if (lr && model.edges[i].from === id) crossSpace = Math.max(crossSpace, height/2 + edgeLabels[i].height + 24);
      }
      cross += crossSpace + 48; thick = Math.max(thick, lr ? width : height);
    }
    let offset = 24;
    for (let i = 0; i < model.edges.length; i += 1) {
      budget.work(); if (ranks.get(model.edges[i].from) !== rank) continue;
      lanes.set(i, major + thick + offset);
      offset += (model.edges[i].label ? (lr ? edgeLabels[i].width : edgeLabels[i].height) : 0) + 24;
    }
    crossMax = Math.max(crossMax, cross); major += thick + offset + 24;
  }
  // Labels occupy a strip outside every node column. Route trunks lie beyond
  // that strip, and their entry/exit legs use the reserved rank intervals.
  const children = [], outer = crossMax + 24, labelRects = [], routes = [];
  let labelCross = 0;
  for (let i = 0; i < model.edges.length; i += 1) {
    budget.work(); if (!model.edges[i].label) continue;
    const label = edgeLabels[i], start = lanes.get(i);
    labelCross = Math.max(labelCross, lr ? label.height : label.width);
    labelRects.push({x:lr ? start : outer, y:lr ? outer : start, width:label.width, height:label.height});
  }
  for (let i = 0; i < model.edges.length; i += 1) {
    budget.work(); const edge = model.edges[i], a = boxes.get(edge.from), b = boxes.get(edge.to), label = edgeLabels[i];
    const start = lr ? [a.x+a.width, a.y+a.height/2] : [a.x+a.width/2, a.y+a.height];
    const end = lr ? [b.x, b.y+b.height/2] : [b.x+b.width/2, b.y];
    const labelStart = lanes.get(i), lane = labelStart + (edge.label ? (lr ? label.width : label.height) : 0) + 8;
    let points = lr ? [start, [lane,start[1]], [lane,end[1]], end] : [start,[start[0],lane],[end[0],lane],end];
    if (edge.label || ranks.get(edge.to) > ranks.get(edge.from)+1) {
      const side = outer + labelCross + 24 + i*16, last = (lr ? b.x : b.y)-12;
      points = lr ? [start,[lane,start[1]],[lane,side],[last,side],[last,end[1]],end] : [start,[start[0],lane],[side,lane],[side,last],[end[0],last],end];
    }
    routes.push(points);
    children.push(...sceneArrow(points, false, budget));
    if (edge.label) children.push(...sceneText(label, lr ? labelStart : outer, lr ? outer : labelStart, budget));
  }
  // Fail closed if a future routing change violates the reserved-label invariant.
  for (const points of routes) for (let i = 1; i < points.length; i += 1) {
    const [x1,y1] = points[i-1], [x2,y2] = points[i];
    for (const rect of labelRects) {
      budget.work();
      if ((x1 === x2 && rect.x < x1 && x1 < rect.x+rect.width && Math.max(y1,y2) > rect.y && Math.min(y1,y2) < rect.y+rect.height) ||
          (y1 === y2 && rect.y < y1 && y1 < rect.y+rect.height && Math.max(x1,x2) > rect.x && Math.min(x1,x2) < rect.x+rect.width)) budget.fail("geometry-limit");
    }
  }
  for (const [id, box] of boxes) {
    budget.work(); children.push(sceneBox(box, nodes.get(id).shape, budget));
    children.push(...sceneText(box.label, box.x+(box.width-box.label.width)/2, box.y+(box.height-box.label.height)/2, budget));
  }
  return {width: Math.max(lr ? major : crossMax, budget.extentX+24), height: Math.max(lr ? crossMax : major, budget.extentY+24), children};
}
