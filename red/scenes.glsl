#version 330
// 《红色》 scene library — one full-screen fragment shader, scene chosen by uScene.
in vec2 uv;
out vec4 fragColor;
uniform int uScene;
uniform float uT;      // global time
uniform float uL;      // time since shot start
uniform vec4 uP;       // per-shot params
uniform vec4 uQ;
uniform vec2 uRes;
uniform sampler2D uAux; // auxiliary texture (chat log, castle mask, ...)

#define PI 3.14159265
float hash11(float p){ p=fract(p*.1031); p*=p+33.33; p*=p+p; return fract(p); }
float hash21(vec2 p){ vec3 p3=fract(vec3(p.xyx)*.1031); p3+=dot(p3,p3.yzx+33.33); return fract((p3.x+p3.y)*p3.z); }
vec2 hash22(vec2 p){ vec3 p3=fract(vec3(p.xyx)*vec3(.1031,.1030,.0973)); p3+=dot(p3,p3.yzx+33.33); return fract((p3.xx+p3.yz)*p3.zy); }
float noise(vec2 p){ vec2 i=floor(p), f=fract(p); f=f*f*(3.-2.*f);
  return mix(mix(hash21(i),hash21(i+vec2(1,0)),f.x), mix(hash21(i+vec2(0,1)),hash21(i+vec2(1,1)),f.x), f.y); }
float fbm(vec2 p){ float v=0., a=.5; for(int i=0;i<5;i++){ v+=a*noise(p); p=mat2(1.6,1.2,-1.2,1.6)*p; a*=.5; } return v; }
mat2 rot(float a){ float c=cos(a), s=sin(a); return mat2(c,-s,s,c); }
vec2 asp(vec2 u){ return (u-.5)*vec2(uRes.x/uRes.y,1.)*2.; }
vec2 vor(vec2 p){ vec2 n=floor(p), f=fract(p); float m1=8., m2=8.;
  for(int j=-1;j<=1;j++) for(int i=-1;i<=1;i++){ vec2 g=vec2(i,j); vec2 o=hash22(n+g); vec2 r=g+o-f; float d=dot(r,r);
    if(d<m1){ m2=m1; m1=d; } else if(d<m2){ m2=d; } }
  return vec2(sqrt(m1), sqrt(m2)-sqrt(m1)); }

// ---------------------------------------------------------------- silk
float silkH(vec2 p, float t){
  vec2 q=p; q.x+=.35*sin(q.y*1.7+t*.6); q.y+=.25*sin(q.x*1.3-t*.4);
  return sin(q.x*2.2+q.y*.9+t*.8)*.6 + sin(q.x*4.1-q.y*1.7-t*1.1)*.25 + sin(q.x*7.3+q.y*3.1+t*.5)*.08;
}
vec3 silk(vec2 u, float t, vec3 base, float sway){
  vec2 p=rot(.45)*asp(u); p.y+=sway*.35*sin(t*.9); p.x+=sway*.2*sin(t*.6+1.);
  float e=.004; float h=silkH(p,t);
  vec2 g=vec2(silkH(p+vec2(e,0.),t)-h, silkH(p+vec2(0.,e),t)-h)/e;
  vec3 n=normalize(vec3(-g*.38,1.));
  vec3 l=normalize(vec3(-.45,.6,.65));
  float diff=max(dot(n,l),0.);
  float spec=pow(max(dot(n,normalize(l+vec3(0,0,1))),0.),70.);
  float sheen=pow(1.-n.z,1.6);
  return base*(.12+.9*diff) + base*sheen*1.4 + vec3(1.,.72,.68)*spec*1.1;
}
vec3 sc_silk(){ return silk(uv, uT*uP.z+uP.w, vec3(.62,.015,.05)*uP.x, uP.y); }

// ---------------------------------------------------------------- wound slit
vec3 sc_slit(){
  vec2 p=asp(uv);
  float w=max(uP.y,1e-3); float xx=p.x/(1.95*w);
  float hh=max(0.,1.-xx*xx);
  float half_=uP.x*.62*pow(hh,.85)+.0025;
  float d=abs(p.y)-half_;
  float inx=step(abs(xx),1.);
  float inside=step(d,0.)*inx;
  vec3 s=silk(uv,uT,vec3(.7,.02,.05),1.);
  float glow=(exp(-max(d,0.)*55.)*.9+exp(-max(d,0.)*9.)*.25)*inx*smoothstep(1.,.85,abs(xx));
  return s*inside*(.6+.6*uP.x) + vec3(1.,.12,.08)*glow*(1.-inside)*(.5+uP.x);
}

// ---------------------------------------------------------------- lava cake
vec3 sc_lava(){
  vec2 p=(asp(uv)-vec2(.6,-.45))*.85;
  float R=.78;
  vec2 q=p*vec2(1.,1.18);
  vec3 col=vec3(.03,0.,.006)+vec3(.35,.03,.02)*exp(-length(p-vec2(0.,.2))*1.6)*(.4+.6*uP.x);
  // plate
  float pl=smoothstep(.012,0.,abs(p.y+.015))*smoothstep(1.5,1.3,abs(p.x));
  col+=vec3(.25,.18,.17)*pl*.5;
  // pool of lava spreading on the plate
  float spread=R+.65*uP.y;
  float pool=smoothstep(.05,0.,abs(p.y+.03)-.025*(1.-abs(p.x)/spread))*smoothstep(spread,spread-.15,abs(p.x));
  float flow=fbm(vec2(p.x*6.-uT*1.5*sign(p.x),p.y*20.));
  col=mix(col, vec3(1.,.22,.02)*(1.2+flow*1.5), pool*uP.y);
  if(p.y>-.02 && length(q)<R){
    float z=sqrt(max(0.,R*R-dot(q,q)));
    vec3 n=normalize(vec3(q.x,q.y,z));
    vec3 l=normalize(vec3(-.5,.75,.55));
    float diff=max(dot(n,l),0.);
    float spec=pow(max(dot(reflect(-l,n),vec3(0,0,1)),0.),26.);
    col=vec3(.075,.028,.016)*(.25+diff*1.1)+spec*.28*vec3(1.,.85,.8);
    col+=vec3(.02,0.,0.)*fbm(n.xy*20.);
    vec2 v=vor(vec2(atan(n.x,n.z)*2.6,n.y*4.6)+3.1);
    float crack=smoothstep(.10*uP.x+.001,0.,v.y)*uP.x;
    float heat=crack*(.75+.25*sin(uT*4.+v.x*12.));
    col+=vec3(1.,.2,.02)*heat*2.6+vec3(1.,.75,.3)*pow(heat,3.)*2.5;
    // ooze trails running down from cracks
    float ooze=smoothstep(.06,0.,abs(fract(atan(n.x,n.z)*3.)-.5)-.4)*smoothstep(.0,.5,1.-n.y)*uP.y;
    col+=vec3(1.,.25,.02)*ooze*1.4;
  }
  return col;
}

// ---------------------------------------------------------------- twirling skirt (top-down)
vec3 sc_skirt(){
  vec2 p=asp(uv)/max(uP.z,.2);
  float r=length(p), a=atan(p.y,p.x);
  float sp=uT*uP.x;
  float fo=sin(a*13.+sp*2.+sin(r*5.-uT*2.)*1.3);
  float rim=.86+.05*sin(a*13.+sp*2.+1.2)+.02*sin(a*31.-sp*3.);
  float m=smoothstep(rim+.006,rim-.006,r)*smoothstep(.06,.1,r);
  vec3 n=normalize(vec3(-sin(a)*fo*.6,cos(a)*fo*.6,1.));
  vec3 l=normalize(vec3(-.4,.5,.75));
  float diff=max(dot(n,l),0.);
  float spec=pow(max(dot(n,normalize(l+vec3(0,0,1))),0.),40.);
  vec3 base=vec3(.66,.02,.06);
  vec3 c=base*(.18+.95*diff)*(.55+.45*smoothstep(0.,.8,r))+spec*vec3(1.,.7,.7)*.8;
  float hem=smoothstep(.03,0.,abs(r-rim));
  c+=vec3(1.,.5,.4)*hem*.35;
  // seeping: streaks of darker red creeping outward from the waist
  float seep=smoothstep(.3,0.,abs(fract(a*6./PI+fbm(vec2(r*3.,uT*.2))*.6)-.5))*smoothstep(uP.y*.9,uP.y*.9-.3,r)*uP.y;
  c=mix(c,vec3(.25,0.,.02),seep*.6);
  vec3 bg=vec3(.02,0.,.005)+vec3(.2,.0,.02)*exp(-r*1.4);
  vec3 col=mix(bg,c,m);
  col=mix(col,vec3(0.),smoothstep(.1,.06,r));
  return col;
}

// ---------------------------------------------------------------- spinel (truncated octahedron), ray-traced with refraction
float sdGem(vec3 p){
  float oc=(abs(p.x)+abs(p.y)+abs(p.z)-1.)*.57735;
  vec3 b=abs(p)-vec3(.74); float bx=length(max(b,0.))+min(max(b.x,max(b.y,b.z)),0.);
  return max(oc,bx);
}
mat3 gemRot(){ float a=uT*.45+uP.x, b=.6+.25*sin(uT*.3);
  mat3 ry=mat3(cos(a),0,-sin(a),0,1,0,sin(a),0,cos(a));
  mat3 rx=mat3(1,0,0,0,cos(b),sin(b),0,-sin(b),cos(b)); return rx*ry; }
vec3 gemN(vec3 p){ vec2 e=vec2(.0008,0.);
  return normalize(vec3(sdGem(p+e.xyy)-sdGem(p-e.xyy), sdGem(p+e.yxy)-sdGem(p-e.yxy), sdGem(p+e.yyx)-sdGem(p-e.yyx))); }
vec3 envL(vec3 d){
  vec3 c=vec3(.06,.0,.012)+vec3(.35,.02,.05)*pow(max(0.,-d.y),2.);
  for(int i=0;i<9;i++){ float fi=float(i); vec3 L=normalize(vec3(sin(fi*2.4),cos(fi*1.7)*.8,sin(fi*1.3+1.)));
    c+=vec3(1.,.9,.85)*smoothstep(.993,.999,dot(d,L))*6.; }
  c+=vec3(1.,.95,.92)*smoothstep(.88,.97,d.y)*2.2;
  c+=vec3(1.,.25,.22)*pow(max(0.,dot(d,normalize(vec3(-.8,.15,.5)))),18.)*3.;
  c+=vec3(1.)*smoothstep(.975,.992,abs(dot(d,normalize(vec3(.6,.35,-.7)))))*3.5;
  c+=vec3(1.,.8,.7)*smoothstep(.96,.99,dot(d,normalize(vec3(.75,-.1,.6))))*2.5;
  return c;
}
vec3 sc_gem(){
  vec2 p=asp(uv)-vec2(.62,.16);
  mat3 R=gemRot();
  vec3 ro=vec3(0.,0.,-3.9), rd=normalize(vec3(p*.55,1.8));
  ro=R*ro; rd=R*rd;
  vec3 bg=vec3(.01,0.,.003)+vec3(.28,.01,.03)*exp(-length(p)*1.1)*uP.y;
  float t=0.; bool hit=false;
  for(int i=0;i<80;i++){ float d=sdGem(ro+rd*t); if(d<.0005){hit=true;break;} t+=d; if(t>8.) break; }
  if(!hit) return bg;
  vec3 pos=ro+rd*t, n=gemN(pos);
  float fr=.08+.92*pow(1.-max(dot(-rd,n),0.),5.);
  vec3 refl=envL(reflect(rd,n));
  vec3 col=vec3(0.);
  float iors[3]; iors[0]=1.70; iors[1]=1.718; iors[2]=1.736;
  for(int c=0;c<3;c++){
    vec3 d=refract(rd,n,1./iors[c]); vec3 q=pos-n*.002; float acc=0.; vec3 outc=vec3(0.);
    for(int b=0;b<3;b++){
      float s=0.;
      for(int i=0;i<40;i++){ float dd=-sdGem(q+d*s); if(dd<.0005) break; s+=max(dd,.002); }
      q+=d*s; acc+=s; vec3 nn=-gemN(q);
      vec3 o=refract(d,nn,iors[c]);
      if(dot(o,o)>.0){ outc=envL(o); break; }
      d=reflect(d,nn); q+=d*.003;
      outc=envL(d)*.3;
    }
    vec3 absorb=exp(-acc*vec3(.12,2.6,1.9));
    col[c]=min((outc*absorb)[c]*1.9,3.);
  }
  col=mix(col,refl,fr)+vec3(.35,0.,.03)*.06;
  return col;
}

// ---------------------------------------------------------------- handkerchief with the chat log
float clothH(vec2 p,float t){ return .05*sin(p.x*3.+t*1.7)+.035*sin(p.y*4.-t*1.3+p.x)+.02*sin((p.x+p.y)*7.+t*2.3); }
vec3 sc_cloth(){
  vec2 p=asp(uv);
  float t=uT;
  vec2 c=(p-vec2(.62,0.))*1.25;
  c=rot(-.06+.03*sin(t*.5))*c;
  float h=clothH(c,t);
  vec2 g=vec2(clothH(c+vec2(.01,0.),t)-h, clothH(c+vec2(0.,.01),t)-h)/.01;
  vec2 w=c+g*.08;
  vec2 tc=w/vec2(1.15,.95)*.5+.5;
  float inside=step(0.,tc.x)*step(tc.x,1.)*step(0.,tc.y)*step(tc.y,1.);
  vec3 n=normalize(vec3(-g*.6,1.));
  float diff=.55+.45*dot(n,normalize(vec3(-.4,.6,.7)));
  vec3 cloth=vec3(.93,.9,.88)*diff;
  float lace=smoothstep(.035,.02,min(min(tc.x,1.-tc.x),min(tc.y,1.-tc.y)));
  float holes=step(.5,fract(atan(tc.y-.5,tc.x-.5)*40./PI))*lace;
  cloth=mix(cloth,cloth*.75,holes*.6);
  vec4 tx=texture(uAux,tc);
  cloth=mix(cloth,tx.rgb*diff,tx.a*(1.-lace));
  vec3 bg=vec3(.08,0.,.01)+vec3(.35,.0,.03)*exp(-length(p)*1.2);
  return mix(bg,cloth,inside);
}

// ---------------------------------------------------------------- tower of Babel
float sdTower(vec3 p){
  float d=1e9;
  for(int k=0;k<9;k++){
    float fk=float(k);
    float r=1.4-fk*.14, y0=fk*.26;
    vec3 q=p-vec3(0.,y0+.13,0.);
    float cyl=max(length(q.xz)-r,abs(q.y)-.13);
    float n=28.-2.*fk; float a=atan(q.z,q.x); float sec=2.*PI/n;
    a=mod(a+sec*.5,sec)-sec*.5; vec2 rq=length(q.xz)*vec2(cos(a),sin(a));
    float arch=max(abs(rq.y)-.032*r,abs(q.y+.02)-.07); arch=max(arch,abs(rq.x-r)-.07);
    cyl=max(cyl,-arch);
    d=min(d,cyl);
  }
  return min(d,p.y+.02);
}
vec3 sc_tower(){
  vec2 p=asp(uv);
  float a=uT*.08+uP.x;
  vec3 ro=vec3(sin(a)*3.0,.18,cos(a)*3.0), ta=vec3(0.,1.1,0.);
  vec3 f=normalize(ta-ro), r=normalize(cross(vec3(0,1,0),f)), u=cross(f,r);
  vec3 rd=normalize(p.x*r+p.y*u+1.25*f);
  vec3 sky=mix(vec3(.55,.06,.03),vec3(.05,0.,.02),smoothstep(-.1,.6,rd.y));
  vec3 sunD=normalize(vec3(sin(a+.9),.3,cos(a+.9)));
  vec3 glowD=normalize(vec3(-sin(a)+.3,.12,-cos(a)));
  sky+=vec3(1.,.45,.2)*pow(max(dot(rd,glowD),0.),60.)*2.+vec3(1.,.2,.05)*pow(max(dot(rd,glowD),0.),6.)*.5;
  float t=0.; bool hit=false;
  for(int i=0;i<90;i++){ float d=sdTower(ro+rd*t); if(d<.001){hit=true;break;} t+=d*.9; if(t>14.) break; }
  if(!hit) return sky;
  vec3 pos=ro+rd*t; vec2 e=vec2(.002,0.);
  vec3 n=normalize(vec3(sdTower(pos+e.xyy)-sdTower(pos-e.xyy),sdTower(pos+e.yxy)-sdTower(pos-e.yxy),sdTower(pos+e.yyx)-sdTower(pos-e.yyx)));
  float diff=max(dot(n,sunD),0.);
  float sh=1.; float st=.02;
  for(int i=0;i<24;i++){ float d=sdTower(pos+sunD*st); sh=min(sh,10.*d/st); st+=clamp(d,.02,.2); if(st>4.) break; }
  sh=clamp(sh,0.,1.);
  vec3 alb=pos.y<-.01?vec3(.12,.03,.02):vec3(.55,.32,.24);
  vec3 col=alb*(vec3(1.,.5,.3)*diff*sh*1.6+vec3(.25,.04,.04));
  col=mix(col,sky*.8,1.-exp(-t*.09));
  return col;
}

// ---------------------------------------------------------------- flesh
float fleshH(vec2 p){ vec2 v=vor(p); return smoothstep(0.,.45,v.x)*-.6+smoothstep(.0,.12,v.y)*.4; }
vec3 sc_flesh(){
  vec2 p=asp(uv)*3.2;
  float beat=uP.x;
  p*=1.-.04*beat;
  p+=vec2(fbm(p*.6+uT*.15),fbm(p*.6-uT*.13))*.5;
  float h=fleshH(p);
  vec2 g=vec2(fleshH(p+vec2(.01,0.))-h,fleshH(p+vec2(0.,.01))-h)/.01;
  vec3 n=normalize(vec3(-g*.25,1.));
  vec3 l=normalize(vec3(-.3,.5,.8));
  float diff=max(dot(n,l),0.);
  float spec=pow(max(dot(n,normalize(l+vec3(0,0,1))),0.),50.);
  vec2 v=vor(p);
  vec3 c=mix(vec3(.45,.02,.05),vec3(.95,.35,.38),smoothstep(.0,.6,v.x));
  c=mix(c,vec3(.12,0.,.02),smoothstep(.06,0.,v.y));
  return c*(.25+.9*diff)+spec*vec3(1.,.85,.85)*.9+vec3(.6,0.,.05)*beat*.25;
}

// ---------------------------------------------------------------- two blobs embracing
float blobF(vec2 p){
  float d=uP.x;
  vec2 c1=vec2(-d,.05*sin(uT)), c2=vec2(d,-.05*sin(uT*1.3));
  c1=rot(uT*.3)*c1; c2=rot(uT*.3)*c2;
  return .16/dot(p-c1,p-c1)+.13/dot(p-c2,p-c2)+.03/dot(p-c1*.3-c2*.3+vec2(0.,.3),p-c1*.3-c2*.3+vec2(0.,.3));
}
vec3 sc_blobs(){
  vec2 p=asp(uv)*.85;
  float f=blobF(p);
  vec3 bg=vec3(.03,0.,.01)+vec3(.25,0.,.03)*exp(-length(p)*1.4);
  if(f<1.) return bg+vec3(.6,.02,.05)*smoothstep(.6,1.,f)*.4;
  vec2 e=vec2(.004,0.);
  vec2 g=vec2(blobF(p+e.xy)-blobF(p-e.xy),blobF(p+e.yx)-blobF(p-e.yx))/(2.*e.x);
  float z=clamp((f-1.)*.5,0.,1.);
  vec3 n=normalize(vec3(-g*.03,sqrt(z)+.15));
  vec3 l=normalize(vec3(-.4,.6,.7));
  float diff=max(dot(n,l),0.);
  float spec=pow(max(dot(reflect(-l,n),vec3(0,0,1)),0.),60.);
  float rim=pow(1.-n.z,3.);
  return vec3(.7,.02,.06)*(.15+diff)+vec3(1.,.4,.4)*rim*.6+spec*1.4;
}

// ---------------------------------------------------------------- 繁星春水 — still water and stars (the only un-red frame)
vec3 stars(vec2 p,float dens){
  vec3 c=vec3(0.);
  for(int k=0;k<3;k++){
    vec2 q=p*(30.+float(k)*24.)+float(k)*17.;
    vec2 i=floor(q), f=fract(q)-.5;
    float h=hash21(i);
    if(h>1.-dens){ vec2 o=(hash22(i)-.5)*.6; float d=length(f-o);
      float tw=.6+.4*sin(uT*(2.+h*5.)+h*40.);
      c+=vec3(.85,.95,1.)*smoothstep(.06,0.,d)*tw*(1.-float(k)*.25); }
  }
  return c;
}
vec3 sc_water(){
  vec2 p=asp(uv);
  float hz=-.05;
  vec3 col;
  if(p.y>hz){
    col=mix(vec3(.05,.2,.24),vec3(.0,.03,.08),smoothstep(hz,1.,p.y));
    col+=stars(p,.12)*1.6;
    col+=vec3(.75,.95,.9)*smoothstep(.09,.085,length(p-vec2(-.9,.55)))+vec3(.3,.6,.6)*exp(-length(p-vec2(-.9,.55))*4.)*.4;
  } else {
    float dy=hz-p.y;
    vec2 rp=vec2(p.x+.012*sin(p.x*20.+uT*1.5)*dy*3.+.006*sin(dy*90.-uT*2.),2.*hz-p.y);
    col=mix(vec3(.05,.2,.24),vec3(.0,.03,.08),smoothstep(hz,1.,rp.y))*.7;
    col+=stars(rp,.12)*.9*(1.-smoothstep(0.,.9,dy));
    col+=vec3(.6,.85,.8)*smoothstep(.04,0.,abs(rp.x+.9+.01*sin(dy*60.-uT*3.)))*exp(-dy*2.)*.5;
    col+=vec3(.15,.35,.35)*pow(max(0.,sin(rp.x*3.+uT*.5)),30.)*.05;
  }
  col+=vec3(.2,.55,.5)*exp(-abs(p.y-hz)*25.)*.15;
  return col*uP.x;
}

// ---------------------------------------------------------------- casino felt
vec3 sc_felt(){
  vec2 p=asp(uv);
  float n=fbm(p*40.)*.5+fbm(p*8.)*.5;
  vec3 c=vec3(.28,.0,.03)*(.6+.4*n);
  float spot=exp(-dot(p-uP.xy,p-uP.xy)*.9);
  return c*(.15+1.3*spot)+vec3(.0);
}

// ---------------------------------------------------------------- night sea, collapsing water, mountains
vec3 sc_sea(){
  vec2 p=asp(uv);
  vec3 sky=mix(vec3(.62,.08,.07),vec3(.02,.01,.05),smoothstep(-.25,.75,p.y));
  vec2 sunp=vec2(.55,.12-uP.y*.25);
  float sd=length(p-sunp);
  sky+=vec3(1.,.25,.1)*smoothstep(.2,.19,sd)*uP.x+vec3(.8,.1,.05)*exp(-sd*4.)*.5*uP.x;
  float m1=.05+.35*fbm(vec2(p.x*1.2+3.,0.))-.1;
  float m2=-.05+.22*fbm(vec2(p.x*2.1+11.,1.));
  vec3 col=sky;
  col=mix(col,vec3(.03,.0,.02),smoothstep(.003,0.,p.y-m1));
  col=mix(col,vec3(.01,.0,.01),smoothstep(.003,0.,p.y-m2));
  float sea=-.18;
  if(p.y<sea){
    float wv=fbm(vec2(p.x*3.,p.y*9.)+vec2(uT*.3,0.));
    col=vec3(.01,.03,.07)*(.6+wv)+vec3(.6,.1,.05)*exp(-abs(p.x-sunp.x)*5.)*smoothstep(.0,.2,sea-p.y)*.3*uP.x*wv;
  }
  // collapsing wall of water
  float crest=.95-uP.z*1.4+.08*sin(p.x*3.+uT)+.06*fbm(vec2(p.x*4.,uT));
  float wall=smoothstep(.01,0.,p.y-crest)*step(-.95,p.y)*uP.w;
  float foam=fbm(vec2(p.x*12.,p.y*4.-uT*3.));
  vec3 wc=vec3(.03,.08,.16)*(.5+foam)+vec3(.8,.9,1.)*smoothstep(.06,0.,abs(p.y-crest))*foam*1.2;
  col=mix(col,wc,wall);
  return col;
}

// ---------------------------------------------------------------- tree rings
vec3 sc_rings(){
  vec2 p=asp(uv)*1.15;
  float r=length(p)+.05*fbm(p*3.)+.02*sin(atan(p.y,p.x)*5.);
  float rr=r*18.*uP.x;
  float ring=smoothstep(.06,0.,abs(fract(rr)-.5)-.44);
  float grow=smoothstep(uP.y+.02,uP.y,r);
  vec3 c=vec3(.05,0.,.005)+vec3(.9,.18,.06)*ring*grow*(.5+.5*fbm(p*9.));
  c+=vec3(1.,.5,.2)*smoothstep(.01,0.,abs(r-uP.y))*1.2;
  return c;
}

// ---------------------------------------------------------------- deep-sea-coloured fire castle
float castle(vec2 p){
  float d=1e9;
  d=min(d,max(abs(p.x)-.55,abs(p.y+.35)-.25));
  for(int i=0;i<5;i++){ float fi=float(i); float x=-.6+fi*.3; float hgt=.25+.18*hash11(fi*3.1)+(i==2?.3:0.);
    d=min(d,max(abs(p.x-x)-.07,abs(p.y+.1-hgt*.5)-hgt*.5-.25));
    vec2 q=p-vec2(x,.15+hgt); d=min(d,max(abs(q.x)*1.6+q.y*.55-.12,-q.y)); }
  return d;
}
vec3 sc_bluefire(){
  vec2 p=asp(uv);
  vec3 col=mix(vec3(0.,.04,.09),vec3(0.,.0,.02),uv.y);
  vec2 cq=p*vec2(1.,1.)+vec2(0.,.15);
  float cs=castle(cq);
  float n=fbm(vec2(p.x*4.,p.y*3.-uT*2.2))*.6+fbm(vec2(p.x*9.,p.y*6.-uT*3.5))*.4;
  float above=smoothstep(-.6,.6,p.y+.35);
  float flame=smoothstep(.22,0.,cs-n*.3*uP.x)*smoothstep(-.02,.05,cs+.02)*(.35+.65*above)*smoothstep(1.,.2,p.y-n*.5);
  vec3 fire=mix(vec3(0.,.12,.45),vec3(.25,.85,1.),flame)*flame*1.3;
  fire+=vec3(.85,1.,1.)*pow(flame,5.)*.8;
  col=mix(col,vec3(0.,.008,.025),smoothstep(.004,0.,cs));
  float win=step(cs,-.02)*step(.7,fract(cq.x*14.))*step(.6,fract(cq.y*9.))*step(.4,hash21(floor(cq*vec2(14.,9.))));
  col+=vec3(.2,.8,1.)*win*.6*uP.x;
  col+=fire;
  // caustics
  vec2 v=vor(p*4.+vec2(uT*.3,uT*.2));
  col+=vec3(0.,.25,.35)*smoothstep(.08,0.,v.y)*.25;
  // baptism: water poured from above
  float s=0.;
  for(int i=0;i<4;i++){ float fi=float(i); float x=-.5+fi*.33+.03*sin(uT+fi);
    s+=smoothstep(.03,0.,abs(p.x-x+.02*sin(p.y*10.+uT*8.)))*smoothstep(1.,-.2,p.y-(1.2-uP.y*2.2)); }
  col+=vec3(.75,.95,1.)*s*uP.y*.9*(.6+.4*fbm(vec2(p.x*30.,p.y*8.+uT*10.)));
  return col;
}

// ---------------------------------------------------------------- tears
vec3 sc_tears(){
  vec2 p=asp(uv);
  vec3 col=vec3(.07,0.,.012)+vec3(.3,0.,.03)*exp(-length(p)*1.3);
  for(int k=0;k<3;k++){
    float sc=6.+float(k)*5.;
    vec2 q=p*sc; q.y+=uT*(2.5+float(k))*uP.x;
    vec2 i=floor(q), f=fract(q)-.5;
    float h=hash21(i+float(k)*31.);
    if(h<uP.y){
      vec2 o=(hash22(i)-.5)*.5; vec2 d=f-o; d.y*=.75; d.y+=d.x*d.x*1.2*sign(d.y);
      float r=length(d);
      float drop=smoothstep(.13,.11,r);
      float hl=smoothstep(.05,0.,length(d-vec2(-.04,.04)));
      col=mix(col,vec3(.9,.25,.3)*(.4+.6*(1.-r*6.)),drop*.85)+vec3(1.)*hl*drop*1.2;
    }
  }
  return col;
}

// ---------------------------------------------------------------- stream of ulcers
vec3 sc_stream(){
  vec2 p=asp(uv);
  vec2 q=p*vec2(1.,2.5); q.x-=uT*.6;
  float f=fbm(q*2.+vec2(fbm(q*1.5+uT*.2),0.)*1.5);
  vec3 col=mix(vec3(.35,0.,.03),vec3(.85,.08,.1),f);
  col+=vec3(1.,.5,.5)*pow(f,6.)*.8;
  vec2 g=q*1.4; vec2 i=floor(g), fr=fract(g)-.5;
  float h=hash21(i);
  if(h>.72){ float r=length(fr-(hash22(i)-.5)*.4);
    col=mix(col,vec3(.98,.88,.7),smoothstep(.14,.1,r)); col=mix(col,vec3(1.,.1,.1),smoothstep(.04,0.,abs(r-.15))*.8); }
  return col*(.55+.45*smoothstep(1.2,0.,abs(p.y)));
}

// ---------------------------------------------------------------- fire (vertical) and heaven fire (radial)
vec3 firePal(float x){ x=clamp(x,0.,1.); return mix(mix(vec3(0.),vec3(.8,.05,.02),smoothstep(0.,.35,x)),mix(vec3(1.,.45,.05),vec3(1.,1.,.85),smoothstep(.7,1.,x)),smoothstep(.35,.7,x)); }
vec3 sc_fire(){
  vec2 p=asp(uv);
  vec2 q=vec2(p.x*2.,p.y*1.5-uT*2.3);
  float n=fbm(q+vec2(fbm(q*1.3+uT*.4),0.))*1.1;
  float h=uP.x*1.6-(p.y+1.)*.8;
  return firePal(n+h-.35)*1.05;
}
vec3 sc_heaven(){
  vec2 p=asp(uv);
  float r=length(p), a=atan(p.y,p.x);
  float n=fbm(vec2(a*3.,r*3.-uT*2.5))+.5*fbm(vec2(a*9.,r*7.-uT*4.));
  float v=uP.x*1.4-r*.9+n*.6;
  vec3 c=firePal(v)*1.4;
  float rays=pow(max(0.,sin(a*24.+fbm(vec2(a*4.,uT))*3.)),8.)*exp(-r*.9)*uP.x;
  c+=vec3(1.,.85,.6)*rays*.7+vec3(1.,1.,.9)*exp(-r*5.)*uP.x*1.5;
  return c;
}

// ---------------------------------------------------------------- smoke
vec3 sc_smoke(){
  vec2 p=asp(uv);
  vec2 q=p*1.4+vec2(0.,-uT*.15);
  vec2 w=vec2(fbm(q+uT*.05),fbm(q+vec2(5.2,1.3)-uT*.04));
  float s=fbm(q+w*2.2);
  float rim=smoothstep(.45,.75,s);
  vec3 c=vec3(.02,0.,.01)+vec3(.55,.03,.06)*rim*(.6+.4*sin(uT*.7))+vec3(1.,.4,.3)*pow(rim,4.)*.5;
  c+=vec3(.9,.1,.1)*exp(-length(p-vec2(-.9,.6))*1.5)*.25;
  return c;
}

// ---------------------------------------------------------------- the red wave
vec3 sc_redwave(){
  vec2 p=asp(uv);
  float prog=uP.x;
  float h=-1.1+prog*2.6+.18*sin(p.x*2.+uT*3.)+.12*fbm(vec2(p.x*3.,uT*2.))-.25*p.x*p.x*(1.-prog);
  float m=smoothstep(.01,-.01,p.y-h);
  float n=fbm(vec2(p.x*3.,p.y*4.-uT*3.));
  vec3 wave=mix(vec3(.55,0.,.04),vec3(1.,.08,.1),n)*(1.+.6*smoothstep(.1,0.,h-p.y));
  wave+=vec3(1.,.6,.5)*smoothstep(.04,0.,abs(p.y-h))*1.2;
  vec3 bg=vec3(.03,0.,.01);
  vec3 c=mix(bg,wave,m);
  return mix(c,vec3(.85,.02,.07)*(.85+.3*n),smoothstep(.8,1.,prog));
}

// ---------------------------------------------------------------- soft red haze
vec3 sc_haze(){
  vec2 p=asp(uv);
  vec3 c=mix(vec3(.75,.15,.2),vec3(.35,.02,.07),smoothstep(-1.,1.,p.y));
  c+=vec3(1.,.6,.6)*exp(-dot(p-vec2(.2*sin(uT*.2),.1),p-vec2(.2*sin(uT*.2),.1))*1.5)*.25;
  c+=vec3(1.)*fbm(p*3.+uT*.05)*.05;
  return c*uP.x;
}

// ---------------------------------------------------------------- eclipse / Ra
vec3 sc_corona(){
  vec2 p=asp(uv);
  float r=length(p), a=atan(p.y,p.x);
  float R=.42;
  float streak=fbm(vec2(a*6.,r*1.5-uT*.1))*.7+fbm(vec2(a*18.,r*3.))*.3;
  float cor=exp(-(r-R)*3.2)*(.4+streak)*step(R,r);
  vec3 c=vec3(1.,.55,.4)*cor*uP.x+vec3(1.,.1,.05)*smoothstep(.025,0.,abs(r-R))*uP.x*1.5;
  c*=smoothstep(R-.002,R+.002,r);
  return c;
}

// ---------------------------------------------------------------- high platform of disgust
float sdAltar(vec3 p){
  float d=p.y+1.;
  for(int k=0;k<6;k++){ float fk=float(k); vec3 b=abs(p-vec3(0.,-1.+fk*.32+.16,0.))-vec3(1.6-fk*.22,.16,1.6-fk*.22);
    d=min(d,length(max(b,0.))+min(max(b.x,max(b.y,b.z)),0.)); }
  vec3 b=abs(p-vec3(0.,1.25,0.))-vec3(.28,.4,.28); d=min(d,length(max(b,0.))+min(max(b.x,max(b.y,b.z)),0.));
  return d;
}
vec3 sc_altar(){
  vec2 p=asp(uv);
  float a=.25+uT*.03;
  vec3 ro=vec3(sin(a)*6.,-.75,cos(a)*6.), ta=vec3(0.,.6,0.);
  vec3 f=normalize(ta-ro), r=normalize(cross(vec3(0,1,0),f)), u=cross(f,r);
  vec3 rd=normalize(p.x*r+p.y*u+1.5*f);
  vec3 sky=mix(vec3(.9,.08,.05),vec3(.08,0.,.02),smoothstep(-.1,.7,rd.y));
  float t=0.; bool hit=false;
  for(int i=0;i<80;i++){ float d=sdAltar(ro+rd*t); if(d<.001){hit=true;break;} t+=d; if(t>20.) break; }
  vec3 col=sky;
  if(hit){ vec3 pos=ro+rd*t; vec2 e=vec2(.002,0.);
    vec3 n=normalize(vec3(sdAltar(pos+e.xyy)-sdAltar(pos-e.xyy),sdAltar(pos+e.yxy)-sdAltar(pos-e.yxy),sdAltar(pos+e.yyx)-sdAltar(pos-e.yyx)));
    float rim=pow(1.-max(dot(n,-rd),0.),3.);
    col=vec3(.02,0.,.01)+vec3(1.,.15,.08)*rim*.6+vec3(.3,.02,.02)*max(n.y,0.)*.3; }
  float gr=pow(max(0.,1.-length(p-vec2(0.,.55))*.8),3.)*.6;
  col+=vec3(1.,.3,.15)*gr*uP.x;
  return col;
}

vec3 sc_gradient(){ return mix(uQ.rgb,uP.rgb,uv.y); }

void main(){
  vec3 c=vec3(0.);
  if(uScene==1) c=sc_silk();
  else if(uScene==2) c=sc_slit();
  else if(uScene==3) c=sc_lava();
  else if(uScene==4) c=sc_skirt();
  else if(uScene==5) c=sc_gem();
  else if(uScene==6) c=sc_cloth();
  else if(uScene==7) c=sc_tower();
  else if(uScene==8) c=sc_flesh();
  else if(uScene==9) c=sc_blobs();
  else if(uScene==10) c=sc_water();
  else if(uScene==11) c=sc_felt();
  else if(uScene==12) c=sc_sea();
  else if(uScene==13) c=sc_rings();
  else if(uScene==14) c=sc_bluefire();
  else if(uScene==15) c=sc_tears();
  else if(uScene==16) c=sc_stream();
  else if(uScene==17) c=sc_fire();
  else if(uScene==18) c=sc_heaven();
  else if(uScene==19) c=sc_smoke();
  else if(uScene==20) c=sc_redwave();
  else if(uScene==21) c=sc_haze();
  else if(uScene==22) c=sc_corona();
  else if(uScene==23) c=sc_altar();
  else if(uScene==24) c=sc_gradient();
  fragColor=vec4(max(c,0.),1.);
}
