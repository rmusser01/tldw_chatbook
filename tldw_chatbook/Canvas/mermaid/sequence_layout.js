function layoutSequence(model, budget) {
  const columns = new Map(), children = [], headers = [];
  // A side note has its own 224px lane on either side of every participant.
  let x = 248, headerHeight = 0;
  for (const participant of model.participants) {
    budget.work(); const label = wrapLabel(participant.label, budget);
    const box = {x, y:24, width: Math.max(64,label.width+32), height:label.height+24, label};
    columns.set(participant.id, x+box.width/2); headers.push(box);
    headerHeight = Math.max(headerHeight, box.height); x += 288;
  }
  let y = 24+headerHeight+32;
  const events = [...model.messages.map(row => ({...row, kind:"message"})), ...model.notes.map(row => ({...row, kind:"note"}))].sort((a,b) => { budget.work(); return a.order-b.order; });
  for (const event of events) {
    budget.work(); const label = wrapLabel(event.label, budget);
    if (event.kind === "message") {
      const a = columns.get(event.from), b = columns.get(event.to);
      children.push(...sceneText(label, Math.min(a,b)+8, y, budget));
      y += label.height+8; children.push(...sceneArrow([[a,y],[b,y]],event.dashed,budget)); y += 32;
    } else {
      const positions = event.participants.map(id => columns.get(id));
      const center = (Math.min(...positions)+Math.max(...positions))/2;
      const span = event.side === "over" ? Math.max(...positions)-Math.min(...positions)+32 : 0;
      const width = Math.max(64,label.width+32,span);
      const left = event.side === "left" ? center-width-16 : event.side === "right" ? center+16 : center-width/2;
      const box = {x:left,y,width,height:label.height+24};
      children.push(sceneBox(box,"rect",budget), ...sceneText(label,left+16,y+12,budget)); y += box.height+24;
    }
  }
  // Lifelines precede opaque note boxes so notes reserve and obscure their lane.
  const lifelines = [];
  for (const center of columns.values()) lifelines.push(sceneNode("path", [["d",`M${center} ${24+headerHeight}V${y}`],["stroke","black"],["stroke-dasharray","4 4"]]));
  const head = [];
  for (const box of headers) head.push(sceneBox(box,"rect",budget), ...sceneText(box.label,box.x+16,box.y+12,budget));
  return {width:x+32,height:y+24,children:[...lifelines,...children,...head]};
}
