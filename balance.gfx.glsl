#version 430
// CREDIT: Template from https://github.com/yosshin4004/minimal_gl/blob/master/examples/01_basic_exe_exportable.gfx.glsl, says "Copyright (C) 2020 Yosshin(@yosshin4004)"

layout(location = 0) uniform int waveOutPosition;
#if defined(EXPORT_EXECUTABLE)
	vec2 resolution = {SCREEN_XRESO, SCREEN_YRESO};
	#define NUM_SAMPLES_PER_SEC 48000.
	float time = waveOutPosition / NUM_SAMPLES_PER_SEC;
#else
	layout(location = 2) uniform float time;
	layout(location = 3) uniform vec2 resolution;
#endif

out vec4 outColor;

// CREDIT: https://github.com/0b5vr/planefiller/blob/fuck/planefiller.snd.glsl
vec2 sincos(float t) {
	return vec2(cos(t), sin(t));
}

mat2 rotate2D(float x) {
	vec2 v = sincos(x);
	return mat2(v.x, v.y, -v.y, v.x);
}

mat3 rotateX(float x) {
	return mat3(rotate2D(x));
}

mat3 R(float x, int o){
    vec2 v=sincos(x);
    int a=o,b=(o+1)%3;
    mat3 m=mat3(1);
    m[a][a]=v.x; m[b][a]=v.y;
    m[a][b]=-v.y; m[b][b]= v.x;
    return m;
}
/* R(t,0)=rotX, R(t,1)=rotY, R(t,2)=rotZ */


// CREDIT: Inspiration from https://www.shadertoy.com/view/3cScWy by Xor
// NOTE: Very sensitive to FOV, should be 1 to match above
//       Varying tunnel radius also interesting
//       Can be changed to sphere by including z in cylinder length
//       Coloring, multiplier on i is interesting
//       Instead of dividing with d on coloring line, can multiply
//       Multiplier on sin in offset can control randomness, going quite extreme
//       Coloring can be given an offset to desaturate a bit
//       Letting rotation depend on time can give more life
vec4 scene0(vec3 p_in, vec3 dir, float time) {
	vec3 p =p_in;
	vec4 O=vec4(0.);
	float d=0, z=0, i=0;
	for (; i<3e1; i++) {
		// Cylinder + Gyroid		
//		d=abs(length(p.xy)-5. + dot(sin(p.xyz), cos(p.yzx)));
		vec3 p_ = R(0.01, 1)*(p*0.9);
		// 0.003 is just the best constant
		d=.003+abs(length(p.xy)-4.+1.0*dot(sin(p_),cos(p_).yzx));
		z+=d;
		O+=(1.0+sin(i*0.3+z+time+vec4(6*sin(time+z*0.2+0.5),1,2*sin(time*0.1 + z*0.9+0.3),0)))/d;
		// O+=(1.+sin(i*0.3+z+time+vec4(6,1,2,0)))/d;
		p += dir*d;
		//p=z*dir;

		for(float f=1.;f++<2.;
					//Blocky, stretched waves
					p+=0.5*sin(round(p.yxz*6.)/3.*f)/f);
	}
	// float c=length(p)*0.01;
	// return vec4(vec3(c), 1.);
	return tanh(O/1e2);
}



// Font parameters, tweaking these is fun :)
// ----

const float baseWidth = 4.0;
const float baseHeight = 4.6;
const float baseHeightUpper = 2.0;

const float verticalSpacingLow = 1.0;
const float verticalSpacingHigh = 0.3;
const float horisontalSpacing = 1.3;

const float letterWidth = 2.0 * baseWidth + horisontalSpacing;

const float letterSpacing = 0.7;

const float rounding = 1.5;


// Helper grid so common coordinates in font can be easily computed
// ----

const float X[4] = float[4](0.0, baseWidth, baseWidth + horisontalSpacing, baseWidth + horisontalSpacing + baseWidth);

const float Y[7] = float[7](0.0, baseHeight, baseHeight + verticalSpacingLow, baseHeight + verticalSpacingLow + baseHeightUpper, baseHeight + verticalSpacingLow + baseHeightUpper + verticalSpacingHigh, baseHeight + verticalSpacingLow + baseHeightUpper + verticalSpacingHigh + baseHeightUpper, 50.);

vec2 G(int x, int y) {
	return vec2(X[x], Y[y]);
}

// Basic signed distance functions
// ----

float sdBox( vec2 p, vec2 b )
{
  vec2 d = abs(p) - b;
  return length(max(d,0.0))
         + min(max(d.x,d.y),0.0); // remove this line for an only partially signed sdf 
}

float sdBox(vec2 p, vec2 ll, vec2 ur)
{
	vec2 d = ur - ll;
	return sdBox(p - ll - d/2.0, d/2.0);
}

float sdRoundBox( vec2 p, vec2 b, float r )
{
  vec2 d = abs(p) - b;
  return length(max(d,0.0)) - r
         + min(max(d.x,d.y),0.0); // remove this line for an only partially signed sdf 
}

float sdRoundBox(vec2 p, vec2 ll, vec2 ur, float r)
{
	vec2 d = ur - ll;
	return sdRoundBox(p - ll - d/2.0, d/2.0 - r, r);
}

// Slope -1 through a
float sdLine2(vec2 p, vec2 a) {
    return dot(p - a, vec2(1.0/sqrt(2.0)));
}


// Letter distance functions composed of simpler SDFs
// ----
float F(vec2 p) {
	float d = sdBox(p, G(0, 0), G(3, 5));
	d = max(d, -sdBox(p, G(1, 0), G(3, 2)));
	return d;
}

float N(vec2 p) {
	float d = sdBox(p, G(0, 0), G(3, 5));

	float up = -sdLine2(p, G(1, 5));
	float down = sdLine2(p, G(2, 2));

	float cut = sdBox(p, G(1, 0), G(2, 5));
	cut = max(cut, min(up, down));
	d = max(d, -cut);

	return d;	
}

float U(vec2 p) {
	float d = sdBox(p, G(0, 0), G(3, 5));
	d = max(d, -sdBox(p, G(1, 1), G(2, 5)));

	// rounding
	float c = sdRoundBox(p, G(0,0), G(3, 6), rounding);
	d = max(d, c);

	return d;   
}

float Q(vec2 p) {
	float d = sdBox(p, G(0, 0), G(3, 5));
	d = max(d, -sdBox(p, G(1, 1), G(2, 3)));

	// rounding
	float c = sdRoundBox(p, G(0,0), G(3, 5), rounding);
	d = max(d, c);

	// bottom protrusion
	d = min(d, sdBox(p, G(2, 0), G(3, 1)));

	return d;
}

float E(vec2 p) {
	float d = sdBox(p, G(0, 0), G(3, 5));
	d = max(d, -sdBox(p, G(1, 1), G(3, 2)));
	return d;
}

float FNUQUE(vec2 p) {
    float d = F(p);
	
	p = p - vec2(letterWidth + letterSpacing, 0.0); // x offset
	d = min(d, N(p));

	p = p - vec2(letterWidth + letterSpacing, 0.0);
	d = min(d, U(p));

	p = p - vec2(letterWidth + letterSpacing, 0.0);
	d = min(d, Q(p));

	p = p - vec2(letterWidth + letterSpacing, 0.0);
	d = min(d, U(p));

	p = p - vec2(letterWidth + letterSpacing, 0.0);
	d = min(d, E(p));

	float BottomY = G(0, 3).y;
	float TopY = G(0, 4).y;
	d = max(d, -sdBox(p, vec2(-150., BottomY), vec2(150., TopY)));

    return d;
}



void main(){
	vec2 uv = (gl_FragCoord.xy*2 - resolution) / resolution.yy;
	// vec3 p = vec3(0.);
	// p.z-=time;

	// vec3 d = normalize(vec3(uv, -1));

	//outColor = vec4(uv, 0, 1);
	// outColor = scene0(p, R(time*0.1, 0)*R(0.3, 1)*d, time);

	float d, q;

    vec2 p = uv*20.0;
    p.x *= 1.0 + p.y/55.0;
    p.x += 28.0;
    d = FNUQUE(p);
    q = smoothstep(0.0, -0.05, d);
    outColor = vec4(vec3(1.3)*q, q);


}

void mainscene0(){
	vec2 uv = (gl_FragCoord.xy*2 - resolution) / resolution.yy;
	vec3 p = vec3(0.);
	p.z-=time;

	vec3 d = normalize(vec3(uv, -1));

	//outColor = vec4(uv, 0, 1);
	outColor = scene0(p, R(time*0.1, 0)*R(0.3, 1)*d, time);
}

/*
    "Accretion" by @XorDev
    
    I discovered an interesting refraction effect
    by adding the raymarch iterator to the turbulence!
    https://x.com/XorDev/status/1936884244128661986
*/

// /*
//     "Accretion" by @XorDev
    
//     I discovered an interesting refraction effect
//     by adding the raymarch iterator to the turbulence!
//     https://x.com/XorDev/status/1936884244128661986
// */

// void mainImage(out vec4 O, vec2 I)
// {
//     //Raymarch depth
//     float z,
//     //Step distance
//     d,
//     //Raymarch iterator
//     i;
//     //Clear fragColor and raymarch 20 steps
//     for(O*=i; i++<2e1; )
//     {
//         //Sample point (from ray direction)
//         vec3 p = z*normalize(vec3(I+I,0)-iResolution.xyx)+.1;
        
//         //Polar coordinates and additional transformations
//         p = vec3(atan(p.y/.2,p.x)*2., p.z/4., length(p.xy)-1.-z*.5);
// //        p += 0.3*sin(p.zyx*3.);
// //           p = round(p.xzy*10.1)/10.1;
        
//         //Apply turbulence and refraction effect
//         for(d=0.; d++<7.;)
//             p += 3.5*sin(round(p.yzx*0.7)/3.*d+0.8*iTime+0.3*cos(d))/d;
            
//         //Distance to cylinder and waves with refraction
//         z += d = length(vec4(.4*cos(p)-.4, p.z));
        
//         //Coloring and brightness
//         O += (1.1+cos(p.x+i*.4+z+vec4(6,1,2,0)))/d;
//     }
//     //Tanh tonemap
//     O = tanh(O*O/4e2);
// }


// /*
//     @yufengjie brought up the idea of distorting a couple spheres
//     instead of a tunnel, so i tinkered with this a bit
    
//     the only changes are there are two vecs, q and p,
//     both are distorted differently in the turbulence loop,
//     then the min of the two spheres is taken
    
//     and the colors are different :D
// */

// void mainImage(out vec4 o, vec2 u) {
    
//     vec3 q,p = iResolution;
    
//     float i, s,
//           // start the ray at a small random distance,
//           // this will reduce banding
//           d = .125*texelFetch(iChannel0, ivec2(u)%1024, 0).a,
//           t = iTime;
          
//     // scale coords
//     u =(u+u-p.xy)/p.y;
    
//     for(o*=i; i++<1e2; ) {
        
//         // shorthand for standard raymarch sample, then move forward:
//         // p = ro + rd * d, p.z -= 5.;
//         q = p = vec3(u * d, d - 5.);

//         // turbulence
//         for (s = 1.; s++ <8.;
//             q += sin(.6*t+p.zxy*s*.3)*.4,
//             p += sin(t+round(p.yzx*1.6)/1.6*s)*.25);

//         // distance to spheres
//         d += s = .005 + abs(-(length(p.xy*2.03+1.)-2. - length(q-1.)-3.))*.2;
        
//         // color: 1.+cos so we don't go negative, cos(d+vec4(6,4,2,0)) samples from the palette
//         // divide by s for form and distance
//         o += (1.+cos(p.z+vec4(6,4,2,0))) / s;
        
//     }
    
//     // tonemap and divide brightness
//     o = tanh(o / 8e3 / max(length(u), .5));
// }