/* bg-animation.js — Floating glowing cells + DNA strands background */
(function () {
  const canvas = document.createElement('canvas');
  canvas.id = 'bio-canvas';
  document.body.insertBefore(canvas, document.body.firstChild);

  const ctx = canvas.getContext('2d');
  let W, H;

  function resize() {
    W = canvas.width = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  /* ── CELLS ── */
  const CELL_COUNT = 11;
  const cells = Array.from({ length: CELL_COUNT }, (_, i) => ({
    x: Math.random() * window.innerWidth,
    y: Math.random() * window.innerHeight,
    r: 30 + Math.random() * 80,
    vx: (Math.random() - 0.5) * 0.35,
    vy: (Math.random() - 0.5) * 0.35,
    phase: Math.random() * Math.PI * 2,
    speed: 0.004 + Math.random() * 0.006,
    bright: 0.55 + Math.random() * 0.45,
  }));

  /* ── DNA STRANDS ── */
  const DNA_COUNT = 2;
  const strands = Array.from({ length: DNA_COUNT }, () => ({
    x: Math.random() * window.innerWidth * 0.35,
    y: -200 + Math.random() * window.innerHeight * 0.3,
    vy: 0.12 + Math.random() * 0.10,
    alpha: 0.18 + Math.random() * 0.18,
    scale: 0.7 + Math.random() * 0.6,
  }));

  /* ── PARTICLES ── */
  const PART_COUNT = 55;
  const particles = Array.from({ length: PART_COUNT }, () => ({
    x: Math.random() * window.innerWidth,
    y: Math.random() * window.innerHeight,
    r: 0.8 + Math.random() * 1.8,
    vx: (Math.random() - 0.5) * 0.18,
    vy: (Math.random() - 0.5) * 0.18,
    a: 0.2 + Math.random() * 0.5,
  }));

  function drawCell(c, t) {
    const pulse = 1 + 0.06 * Math.sin(t * c.speed * 120 + c.phase);
    const R = c.r * pulse;

    /* outer glow */
    const glow = ctx.createRadialGradient(c.x, c.y, R * 0.5, c.x, c.y, R * 2.2);
    glow.addColorStop(0, `rgba(30,120,220,${0.18 * c.bright})`);
    glow.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.beginPath();
    ctx.arc(c.x, c.y, R * 2.2, 0, Math.PI * 2);
    ctx.fillStyle = glow;
    ctx.fill();

    /* cell body gradient */
    const body = ctx.createRadialGradient(c.x - R * 0.28, c.y - R * 0.28, R * 0.05, c.x, c.y, R);
    body.addColorStop(0, `rgba(14, 60, 130, ${0.82 * c.bright})`);
    body.addColorStop(0.6, `rgba(5, 30, 80,  ${0.90 * c.bright})`);
    body.addColorStop(1, `rgba(2, 10, 40,  ${0.95 * c.bright})`);
    ctx.beginPath();
    ctx.arc(c.x, c.y, R, 0, Math.PI * 2);
    ctx.fillStyle = body;
    ctx.fill();

    /* textured hex-like surface lines */
    const SEGS = 9;
    ctx.save();
    ctx.beginPath();
    ctx.arc(c.x, c.y, R, 0, Math.PI * 2);
    ctx.clip();
    ctx.strokeStyle = `rgba(50, 160, 255, ${0.22 * c.bright})`;
    ctx.lineWidth = 0.6;
    for (let i = 0; i < SEGS; i++) {
      const angle = (i / SEGS) * Math.PI * 2 + t * c.speed * 8;
      const x1 = c.x + Math.cos(angle) * R * 0.2;
      const y1 = c.y + Math.sin(angle) * R * 0.2;
      const x2 = c.x + Math.cos(angle) * R;
      const y2 = c.y + Math.sin(angle) * R;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }
    /* inner concentric rings */
    for (let ri = 0.3; ri < 1; ri += 0.22) {
      ctx.beginPath();
      ctx.arc(c.x, c.y, R * ri, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(70,180,255,${0.13 * c.bright})`;
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }
    ctx.restore();

    /* bright rim */
    ctx.beginPath();
    ctx.arc(c.x, c.y, R, 0, Math.PI * 2);
    ctx.strokeStyle = `rgba(80, 190, 255, ${0.55 * c.bright})`;
    ctx.lineWidth = 1.2;
    ctx.stroke();

    /* specular highlight */
    const shine = ctx.createRadialGradient(
      c.x - R * 0.35, c.y - R * 0.35, 0,
      c.x - R * 0.35, c.y - R * 0.35, R * 0.55
    );
    shine.addColorStop(0, `rgba(160,220,255,${0.22 * c.bright})`);
    shine.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.beginPath();
    ctx.arc(c.x, c.y, R, 0, Math.PI * 2);
    ctx.fillStyle = shine;
    ctx.fill();
  }

  function drawDNA(s, t) {
    const H_SEG = 22 * s.scale;
    const AMP = 28 * s.scale;
    const segments = 28;
    const totalH = segments * H_SEG;
    const cx = s.x;

    ctx.save();
    ctx.globalAlpha = s.alpha;

    for (let i = 0; i < segments; i++) {
      const y = s.y + i * H_SEG;
      const phase = (i / segments) * Math.PI * 4 + t * 0.4;
      const x1 = cx + Math.cos(phase) * AMP;
      const x2 = cx - Math.cos(phase) * AMP;

      /* left strand */
      if (i > 0) {
        const py = s.y + (i - 1) * H_SEG;
        const pp = ((i - 1) / segments) * Math.PI * 4 + t * 0.4;
        ctx.beginPath();
        ctx.moveTo(cx + Math.cos(pp) * AMP, py);
        ctx.lineTo(x1, y);
        ctx.strokeStyle = 'rgba(60,160,255,0.7)';
        ctx.lineWidth = 1.2 * s.scale;
        ctx.stroke();

        /* right strand */
        ctx.beginPath();
        ctx.moveTo(cx - Math.cos(pp) * AMP, py);
        ctx.lineTo(x2, y);
        ctx.stroke();
      }

      /* rungs every 2 segments */
      if (i % 2 === 0) {
        ctx.beginPath();
        ctx.moveTo(x1, y);
        ctx.lineTo(x2, y);
        ctx.strokeStyle = 'rgba(100,200,255,0.45)';
        ctx.lineWidth = 0.8 * s.scale;
        ctx.stroke();

        /* node dots */
        ctx.beginPath();
        ctx.arc(x1, y, 2.2 * s.scale, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(130,210,255,0.9)';
        ctx.fill();
        ctx.beginPath();
        ctx.arc(x2, y, 2.2 * s.scale, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    /* molecule dots scattered around strand */
    const molPositions = [
      { ox: 55 * s.scale, oy: totalH * 0.2 },
      { ox: -50 * s.scale, oy: totalH * 0.55 },
      { ox: 70 * s.scale, oy: totalH * 0.75 },
    ];
    molPositions.forEach(m => {
      const mx = cx + m.ox, my = s.y + m.oy;
      ctx.beginPath();
      ctx.arc(mx, my, 3.5 * s.scale, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(80,180,255,0.5)';
      ctx.fill();
      ctx.beginPath();
      ctx.moveTo(cx, my);
      ctx.lineTo(mx, my);
      ctx.strokeStyle = 'rgba(80,180,255,0.3)';
      ctx.lineWidth = 0.8;
      ctx.stroke();
      [[mx + 12 * s.scale, my - 10 * s.scale], [mx + 18 * s.scale, my + 8 * s.scale]].forEach(([bx, by]) => {
        ctx.beginPath();
        ctx.arc(bx, by, 2 * s.scale, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(80,180,255,0.4)';
        ctx.fill();
        ctx.beginPath();
        ctx.moveTo(mx, my);
        ctx.lineTo(bx, by);
        ctx.strokeStyle = 'rgba(80,180,255,0.2)';
        ctx.lineWidth = 0.6;
        ctx.stroke();
      });
    });

    ctx.restore();
  }

  function drawConnectors() {
    for (let i = 0; i < cells.length; i++) {
      for (let j = i + 1; j < cells.length; j++) {
        const dx = cells[j].x - cells[i].x;
        const dy = cells[j].y - cells[i].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 260) {
          const alpha = (1 - dist / 260) * 0.12;
          ctx.beginPath();
          ctx.moveTo(cells[i].x, cells[i].y);
          ctx.lineTo(cells[j].x, cells[j].y);
          ctx.strokeStyle = `rgba(60,160,255,${alpha})`;
          ctx.lineWidth = 0.6;
          ctx.stroke();
        }
      }
    }
  }

  function loop(t) {
    /* deep navy background */
    ctx.fillStyle = '#020c1e';
    ctx.fillRect(0, 0, W, H);

    /* subtle vignette */
    const vig = ctx.createRadialGradient(W / 2, H / 2, H * 0.25, W / 2, H / 2, H * 0.85);
    vig.addColorStop(0, 'rgba(0,0,0,0)');
    vig.addColorStop(1, 'rgba(0,0,0,0.55)');
    ctx.fillStyle = vig;
    ctx.fillRect(0, 0, W, H);

    const ts = t * 0.001;

    /* particles */
    particles.forEach(p => {
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0) p.x = W; if (p.x > W) p.x = 0;
      if (p.y < 0) p.y = H; if (p.y > H) p.y = 0;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(100,200,255,${p.a * 0.5})`;
      ctx.fill();
    });

    /* DNA strands */
    strands.forEach(s => {
      s.y += s.vy;
      if (s.y > H + 300) s.y = -600;
      drawDNA(s, ts);
    });

    /* connector lines between cells */
    drawConnectors();

    /* cells */
    cells.forEach(c => {
      c.x += c.vx; c.y += c.vy;
      if (c.x < -c.r * 2) c.x = W + c.r;
      if (c.x > W + c.r * 2) c.x = -c.r;
      if (c.y < -c.r * 2) c.y = H + c.r;
      if (c.y > H + c.r * 2) c.y = -c.r;
      drawCell(c, ts);
    });

    requestAnimationFrame(loop);
  }

  requestAnimationFrame(loop);
})();
