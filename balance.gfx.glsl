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

void main(){
	vec2 uv = (gl_FragCoord.xy*2 - resolution) / resolution.yy;
	vec3 p = vec3(0.);
	p.z-=time;

	vec3 d = normalize(vec3(uv, -1));

	//outColor = vec4(uv, 0, 1);
	outColor = scene0(p, R(time*0.1, 0)*R(0.3, 1)*d, time);
}

