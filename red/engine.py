"""Minimal off-screen GL pipeline: scene passes -> composite (overlay, text, shock, mosaic, cracks) -> bloom -> grade."""
import os

import moderngl
import numpy as np

from common import W, H

HERE = os.path.dirname(os.path.abspath(__file__))
VS = """#version 330
in vec2 p; out vec2 uv; void main(){ uv=p*.5+.5; gl_Position=vec4(p,0.,1.); }"""

COMP = """#version 330
in vec2 uv; out vec4 o;
uniform sampler2D A, B, Over, Glow, Text, Crack;
uniform float glowGain;
uniform float mixAB, zoom, pix, crack, wipe, aspect;
uniform vec2 sA, sB;
uniform vec2 shake, res;
uniform vec4 shock, flash;
void main(){
  vec2 c=(uv-.5)/zoom+.5+shake;
  vec2 d=c-shock.xy; d.x*=aspect; float r=length(d);
  float w=shock.w*exp(-pow((r-shock.z)*9.,2.));
  c-=normalize(d+1e-6)*w/vec2(aspect,1.);
  vec2 cp=c;
  if(pix>1.) cp=(floor(c*res/pix)+.5)*pix/res;
  vec2 cc=clamp(cp,.001,.999); vec3 a=texture(A,cc*sA).rgb, b=texture(B,cc*sB).rgb;
  float m=mixAB;
  if(wipe>0.){ float e=uv.x*.85+uv.y*.15; m=smoothstep(mixAB-.03,mixAB+.03,1.-e); }
  vec3 s=mix(a,b,m);
  vec4 ov=texture(Over,cp); s=s*(1.-ov.a)+ov.rgb;
  s+=texture(Glow,cp).rgb*glowGain;
  float k=texture(Crack,c).r*crack;
  s=s*(1.-.55*k)+vec3(1.,.85,.85)*k*.35;
  vec4 tx=texture(Text,uv+shake*.35); s=s*(1.-tx.a)+tx.rgb;
  s=mix(s,flash.rgb,flash.a);
  o=vec4(s,1.);
}"""

BRIGHT = """#version 330
in vec2 uv; out vec4 o; uniform sampler2D S; uniform vec2 px; uniform float thr;
void main(){ vec3 c=vec3(0.);
  for(int j=-1;j<=2;j++) for(int i=-1;i<=2;i++) c+=texture(S,uv+vec2(i,j)*px).rgb;
  c/=16.; o=vec4(max(c-thr,0.),1.); }"""

BLUR = """#version 330
in vec2 uv; out vec4 o; uniform sampler2D S; uniform vec2 dir;
void main(){ float w[5]=float[](.227027,.1945946,.1216216,.054054,.016216);
  vec3 c=texture(S,uv).rgb*w[0];
  for(int i=1;i<5;i++){ c+=texture(S,uv+dir*float(i)).rgb*w[i]; c+=texture(S,uv-dir*float(i)).rgb*w[i]; }
  o=vec4(c,1.); }"""

FINAL = """#version 330
in vec2 uv; out vec4 o;
uniform sampler2D S, B1, B2, Grain;
uniform float ca, bloom, grain, glitch, time, vig, gradeRed, gain, invert;
uniform vec2 res;
float h(float x){ return fract(sin(x*91.7)*43758.5453); }
void main(){
  vec2 u=uv;
  if(glitch>0.){ float band=floor(u.y*48.+floor(time*24.)*7.); float r=h(band);
    if(r<glitch*.6) u.x+=(h(band+3.)-.5)*.12*glitch; }
  vec2 d=u-.5;
  float k=ca*.0016;
  vec3 c=vec3(texture(S,u-d*k).r, texture(S,u).g, texture(S,u+d*k).b);
  if(glitch>0.){ c.r=mix(c.r,texture(S,u+vec2(.012*glitch,0.)).r,glitch); }
  c+=(texture(B1,u).rgb*.9+texture(B2,u).rgb*.7)*bloom;
  c*=gain;
  float l=dot(c,vec3(.3,.59,.11)); c=mix(c,vec3(l,l*.08,l*.12)*1.25,gradeRed);
  c=c/(1.+max(c-.75,0.)*1.15);
  c*=1.-vig*pow(length(d*vec2(1.,.85))*1.25,2.4);
  c=mix(c,1.-c,invert);
  c+=(texture(Grain,uv).r-.5)*grain;
  o=vec4(clamp(c,0.,1.),1.);
}"""


class Engine:
    def __init__(self):
        self.ctx = moderngl.create_standalone_context(backend="egl")
        src = open(os.path.join(HERE, "scenes.glsl")).read()
        self.p_scene = self.ctx.program(vertex_shader=VS, fragment_shader=src)
        self.p_comp = self.ctx.program(vertex_shader=VS, fragment_shader=COMP)
        self.p_bright = self.ctx.program(vertex_shader=VS, fragment_shader=BRIGHT)
        self.p_blur = self.ctx.program(vertex_shader=VS, fragment_shader=BLUR)
        self.p_final = self.ctx.program(vertex_shader=VS, fragment_shader=FINAL)
        quad = self.ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], "f4"))
        self.vaos = {k: self.ctx.simple_vertex_array(p, quad, "p") for k, p in
                     dict(scene=self.p_scene, comp=self.p_comp, bright=self.p_bright, blur=self.p_blur,
                          final=self.p_final).items()}
        t = lambda w, h, c=4, dt="f2": self.ctx.texture((w, h), c, dtype=dt)
        self.tA, self.tB = t(W, H), t(W, H)
        self.fA, self.fB = self.ctx.framebuffer([self.tA]), self.ctx.framebuffer([self.tB])
        self.tOver, self.tText = t(W, H, 4, "f1"), t(W, H, 4, "f4")
        self.tGlow = t(W, H, 4, "f1")
        self.tAux = t(1024, 1024, 4, "f4")
        self.tCrack = t(W, H, 1, "f4")
        self.tComp = t(W, H)
        self.fComp = self.ctx.framebuffer([self.tComp])
        self.q = [t(W // 4, H // 4), t(W // 4, H // 4)]
        self.fq = [self.ctx.framebuffer([x]) for x in self.q]
        self.e = [t(W // 8, H // 8), t(W // 8, H // 8)]
        self.fe = [self.ctx.framebuffer([x]) for x in self.e]
        self.tGrain = t(W, H, 1, "f4")
        self.fOut = self.ctx.simple_framebuffer((W, H), components=3)
        for tex in [self.tA, self.tB, self.tOver, self.tGlow, self.tText, self.tAux, self.tCrack, self.tComp, self.tGrain] + self.q + self.e:
            tex.repeat_x = tex.repeat_y = False
        g = np.random.default_rng(5).random((H, W)).astype("f4")
        self.tGrain.write(g.tobytes())
        self.zero = np.zeros((H, W, 4), "f4")
        self.tOver.write(np.zeros((H, W, 4), np.uint8).tobytes())
        self.tGlow.write(np.zeros((H, W, 4), np.uint8).tobytes())
        self.tText.write(self.zero.tobytes())
        self.tCrack.write(np.zeros((H, W), "f4").tobytes())

    def set_uniforms(self, prog, **kw):
        for k, v in kw.items():
            if k in prog:
                prog[k].value = v

    def scene(self, which, sid, T, L, P=(0, 0, 0, 0), Q=(0, 0, 0, 0), scale=1.0):
        fb = self.fA if which == 0 else self.fB
        fb.use()
        w, h = int(W * scale), int(H * scale)
        self.ctx.viewport = (0, 0, w, h)
        pr = self.p_scene
        self.tAux.use(0)
        self.set_uniforms(pr, uScene=int(sid), uT=float(T), uL=float(L), uP=tuple(map(float, P)),
                          uQ=tuple(map(float, Q)), uRes=(float(w), float(h)), uAux=0)
        self.vaos["scene"].render(moderngl.TRIANGLE_STRIP)
        self.ctx.viewport = (0, 0, W, H)
        return (w / W, h / H)

    def upload(self, over=None, glow=None, text=None, aux=None, crack=None):
        if over is not None:
            self.tOver.write(np.ascontiguousarray(over[::-1]).tobytes())
        if glow is not None:
            self.tGlow.write(np.ascontiguousarray(glow[::-1]).tobytes())
        if text is not None:
            self.tText.write(np.ascontiguousarray(text[::-1]).tobytes())
        if aux is not None:
            self.tAux.write(np.ascontiguousarray(aux[::-1]).astype("f4").tobytes())
        if crack is not None:
            self.tCrack.write(np.ascontiguousarray(crack[::-1]).astype("f4").tobytes())

    def frame(self, scaleA, scaleB, mixAB=0.0, wipe=0.0, zoom=1.0, shake=(0, 0), pix=0.0, crack=0.0,
              shock=(0.5, 0.5, 0.0, 0.0), flash=(1, 1, 1, 0), ca=1.0, bloom=0.6, grain=0.035, glitch=0.0,
              time=0.0, vig=0.35, gradeRed=0.0, gain=1.0, invert=0.0, thr=0.55, glowGain=2.5):
        # composite
        self.fComp.use()
        self.ctx.viewport = (0, 0, W, H)
        pc = self.p_comp
        for i, tex in enumerate([self.tA, self.tB, self.tOver, self.tText, self.tCrack, self.tGlow]):
            tex.use(i)
        self.set_uniforms(pc, A=0, B=1, Over=2, Text=3, Crack=4, Glow=5, glowGain=float(glowGain), mixAB=float(mixAB), zoom=float(zoom),
                          pix=float(pix), crack=float(crack), wipe=float(wipe), aspect=W / H,
                          shake=tuple(map(float, shake)), res=(float(W), float(H)),
                          shock=tuple(map(float, shock)), flash=tuple(map(float, flash)))
        # scenes rendered at reduced scale occupy the lower-left part of their texture
        self.set_uniforms(pc, sA=tuple(map(float, scaleA)), sB=tuple(map(float, scaleB)))
        self.vaos["comp"].render(moderngl.TRIANGLE_STRIP)
        # bloom
        self.fq[0].use()
        self.ctx.viewport = (0, 0, W // 4, H // 4)
        self.tComp.use(0)
        self.set_uniforms(self.p_bright, S=0, px=(1.0 / W, 1.0 / H), thr=float(thr))
        self.vaos["bright"].render(moderngl.TRIANGLE_STRIP)
        for _ in range(2):
            self._blur(self.q, self.fq, W // 4, H // 4, 1.5)
        self.fe[0].use()
        self.ctx.viewport = (0, 0, W // 8, H // 8)
        self.q[0].use(0)
        self.set_uniforms(self.p_bright, S=0, px=(4.0 / W, 4.0 / H), thr=0.0)
        self.vaos["bright"].render(moderngl.TRIANGLE_STRIP)
        for _ in range(2):
            self._blur(self.e, self.fe, W // 8, H // 8, 2.0)
        # final
        self.fOut.use()
        self.ctx.viewport = (0, 0, W, H)
        self.tComp.use(0)
        self.q[0].use(1)
        self.e[0].use(2)
        self.tGrain.use(3)
        self.set_uniforms(self.p_final, S=0, B1=1, B2=2, Grain=3, ca=float(ca), bloom=float(bloom), grain=float(grain),
                          glitch=float(glitch), time=float(time), vig=float(vig), gradeRed=float(gradeRed),
                          gain=float(gain), invert=float(invert), res=(float(W), float(H)))
        self.vaos["final"].render(moderngl.TRIANGLE_STRIP)
        data = self.fOut.read(components=3)
        return np.frombuffer(data, np.uint8).reshape(H, W, 3)[::-1]

    def _blur(self, texs, fbs, w, h, spread):
        self.ctx.viewport = (0, 0, w, h)
        fbs[1].use()
        texs[0].use(0)
        self.set_uniforms(self.p_blur, S=0, dir=(spread / w, 0.0))
        self.vaos["blur"].render(moderngl.TRIANGLE_STRIP)
        fbs[0].use()
        texs[1].use(0)
        self.set_uniforms(self.p_blur, S=0, dir=(0.0, spread / h))
        self.vaos["blur"].render(moderngl.TRIANGLE_STRIP)
