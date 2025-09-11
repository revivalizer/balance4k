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

void main(){
	vec2 uv = gl_FragCoord.xy / resolution;
	outColor = vec4(uv + sin(time), 0, 1);
}

