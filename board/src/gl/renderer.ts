// WebGL2 renderer: a paper ground and one pass of stippled ink dots.
import type { Ink } from '../util';

const STRIDE = 16; // x f32, y f32, size f32, rgba u8x4

export class Dots {
  n = 0;
  cap: number;
  // camera, for culling dots that fall outside the viewport
  private k = 1;
  private tx = 0;
  private ty = 0;
  private vw = 1e9;
  private vh = 1e9;
  buf: ArrayBuffer;
  f32: Float32Array;
  u8: Uint8Array;

  constructor(cap = 1 << 17) {
    this.cap = cap;
    this.buf = new ArrayBuffer(cap * STRIDE);
    this.f32 = new Float32Array(this.buf);
    this.u8 = new Uint8Array(this.buf);
  }

  reset(): void {
    this.n = 0;
  }

  view(k: number, tx: number, ty: number, w: number, h: number): void {
    this.k = k;
    this.tx = tx;
    this.ty = ty;
    this.vw = w;
    this.vh = h;
  }

  private grow(): void {
    const cap = this.cap * 2;
    const buf = new ArrayBuffer(cap * STRIDE);
    new Uint8Array(buf).set(this.u8);
    this.cap = cap;
    this.buf = buf;
    this.f32 = new Float32Array(buf);
    this.u8 = new Uint8Array(buf);
  }

  /** Add a crisp square (memory-map cells). Encoded as a negative size, so the vertex layout is unchanged. */
  addSquare(x: number, y: number, size: number, ink: Ink, alpha: number): void {
    this.add(x, y, -Math.abs(size), ink, alpha);
  }

  /** Add one dot. x,y in CSS px; size = diameter in CSS px (negative = square); alpha 0..1. */
  add(x: number, y: number, size: number, ink: Ink, alpha: number): void {
    if (alpha <= 0.004) return;
    const sx = x * this.k + this.tx;
    const sy = y * this.k + this.ty;
    if (sx < -24 || sy < -24 || sx > this.vw + 24 || sy > this.vh + 24) return;
    if (this.n >= this.cap) {
      if (this.cap >= 1 << 20) return;
      this.grow();
    }
    const i = this.n++;
    const f = i * 4;
    this.f32[f] = x;
    this.f32[f + 1] = y;
    this.f32[f + 2] = size;
    const b = i * STRIDE + 12;
    this.u8[b] = ink[0];
    this.u8[b + 1] = ink[1];
    this.u8[b + 2] = ink[2];
    this.u8[b + 3] = alpha >= 1 ? 255 : (alpha * 255) | 0;
  }
}

const PAPER_VS = `#version 300 es
void main(){
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const PAPER_FS = `#version 300 es
precision highp float;
uniform vec2 u_res;
uniform float u_dpr;
uniform vec3 u_base;
uniform float u_dark;
out vec4 o;
float hash(vec2 p){ p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
float noise(vec2 p){
  vec2 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1,0)), f.x), mix(hash(i + vec2(0,1)), hash(i + vec2(1,1)), f.x), f.y);
}
void main(){
  vec2 px = gl_FragCoord.xy / u_dpr;
  float grain = hash(floor(gl_FragCoord.xy)) - 0.5;
  float fib = noise(vec2(px.x * 0.015, px.y * 0.6)) - 0.5;
  float cloud = noise(px * 0.0035) + 0.5 * noise(px * 0.011) - 0.75;
  vec2 uv = gl_FragCoord.xy / u_res;
  float vig = smoothstep(1.05, 0.25, length((uv - 0.5) * vec2(1.0, 1.25)));
  vec3 c = u_base + grain * 0.022 + fib * 0.014 + cloud * 0.03;
  c *= mix(mix(0.93, 0.80, u_dark), 1.0, vig);
  o = vec4(c, 1.0);
}`;

const DOT_VS = `#version 300 es
in vec2 a_pos;
in float a_size;
in vec4 a_col;
uniform vec2 u_css;
uniform float u_dpr;
uniform vec3 u_cam; // zoom k, translate x, translate y (CSS px)
out vec4 v_col;
out float v_seed;
out float v_sq;
void main(){
  vec2 p = (a_pos * u_cam.x + u_cam.yz) / u_css * 2.0 - 1.0;
  gl_Position = vec4(p.x, -p.y, 0.0, 1.0);
  // dots grow slower than the spacing between them, so zooming reveals detail instead of blobs
  gl_PointSize = max(1.0, abs(a_size) * u_dpr * pow(u_cam.x, 0.55));
  v_sq = a_size < 0.0 ? 1.0 : 0.0;
  v_col = a_col;
  v_seed = fract(sin(float(gl_VertexID) * 12.9898) * 43758.5453);
}`;

// Ink dot with an irregular, slightly bled edge; premultiplied output.
const DOT_FS = `#version 300 es
precision mediump float;
in vec4 v_col;
in float v_seed;
in float v_sq;
out vec4 o;
void main(){
  vec2 c = gl_PointCoord * 2.0 - 1.0;
  if (v_sq > 0.5) {
    // crisp square with a faint ink grain (memory-map cells)
    float e = max(abs(c.x), abs(c.y));
    float g = 0.92 + 0.08 * fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
    float sa = smoothstep(1.0, 0.88, e) * v_col.a * g;
    if (sa < 0.003) discard;
    o = vec4(v_col.rgb * sa, sa);
    return;
  }
  float ang = atan(c.y, c.x);
  float rr = 1.0 + 0.13 * sin(ang * 3.0 + v_seed * 6.283) + 0.07 * sin(ang * 5.0 + v_seed * 17.0);
  float d = length(c) * rr;
  float a = smoothstep(1.0, 0.45, d) * v_col.a;
  if (a < 0.003) discard;
  o = vec4(v_col.rgb * a, a);
}`;

function compile(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const s = gl.createShader(type)!;
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? 'shader error');
  return s;
}

function program(gl: WebGL2RenderingContext, vs: string, fs: string): WebGLProgram {
  const p = gl.createProgram()!;
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) ?? 'link error');
  return p;
}

export class Renderer {
  gl: WebGL2RenderingContext;
  canvas: HTMLCanvasElement;
  dpr = 1;
  cssW = 1;
  cssH = 1;
  private paper: WebGLProgram;
  private dots: WebGLProgram;
  private vao: WebGLVertexArrayObject;
  private vbo: WebGLBuffer;
  private vboCap = 0;
  base: [number, number, number] = [0.953, 0.933, 0.89];
  dark = 0;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const gl = canvas.getContext('webgl2', { antialias: false, alpha: false, premultipliedAlpha: true, preserveDrawingBuffer: false });
    if (!gl) throw new Error('WebGL2 is not available in this browser');
    this.gl = gl;
    this.paper = program(gl, PAPER_VS, PAPER_FS);
    this.dots = program(gl, DOT_VS, DOT_FS);
    this.vao = gl.createVertexArray()!;
    this.vbo = gl.createBuffer()!;
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
    const loc = (n: string) => gl.getAttribLocation(this.dots, n);
    gl.enableVertexAttribArray(loc('a_pos'));
    gl.vertexAttribPointer(loc('a_pos'), 2, gl.FLOAT, false, STRIDE, 0);
    gl.enableVertexAttribArray(loc('a_size'));
    gl.vertexAttribPointer(loc('a_size'), 1, gl.FLOAT, false, STRIDE, 8);
    gl.enableVertexAttribArray(loc('a_col'));
    gl.vertexAttribPointer(loc('a_col'), 4, gl.UNSIGNED_BYTE, true, STRIDE, 12);
    gl.bindVertexArray(null);
  }

  resize(): boolean {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(window.innerWidth));
    const h = Math.max(1, Math.floor(window.innerHeight));
    if (w === this.cssW && h === this.cssH && dpr === this.dpr) return false;
    this.cssW = w;
    this.cssH = h;
    this.dpr = dpr;
    this.canvas.width = Math.floor(w * dpr);
    this.canvas.height = Math.floor(h * dpr);
    this.canvas.style.width = `${w}px`;
    this.canvas.style.height = `${h}px`;
    return true;
  }

  cam: [number, number, number] = [1, 0, 0];

  render(d: Dots): void {
    const gl = this.gl;
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.disable(gl.BLEND);
    gl.useProgram(this.paper);
    gl.uniform2f(gl.getUniformLocation(this.paper, 'u_res'), this.canvas.width, this.canvas.height);
    gl.uniform1f(gl.getUniformLocation(this.paper, 'u_dpr'), this.dpr);
    gl.uniform3f(gl.getUniformLocation(this.paper, 'u_base'), ...this.base);
    gl.uniform1f(gl.getUniformLocation(this.paper, 'u_dark'), this.dark);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    if (!d.n) return;
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(this.dots);
    gl.uniform2f(gl.getUniformLocation(this.dots, 'u_css'), this.cssW, this.cssH);
    gl.uniform1f(gl.getUniformLocation(this.dots, 'u_dpr'), this.dpr);
    gl.uniform3f(gl.getUniformLocation(this.dots, 'u_cam'), ...this.cam);
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
    const bytes = d.n * STRIDE;
    if (d.cap * STRIDE > this.vboCap) {
      this.vboCap = d.cap * STRIDE;
      gl.bufferData(gl.ARRAY_BUFFER, this.vboCap, gl.DYNAMIC_DRAW);
    }
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, new Uint8Array(d.buf, 0, bytes));
    gl.drawArrays(gl.POINTS, 0, d.n);
    gl.bindVertexArray(null);
  }
}
