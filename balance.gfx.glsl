 #version 430
// CREDIT: Template from https://github.com/yosshin4004/minimal_gl/blob/master/examples/01_basic_exe_exportable.gfx.glsl, says "Copyright (C) 2020 Yosshin(@yosshin4004)"

// CREDIT: FROM brainfiller/Ob5vr (https://github.com/0b5vr/brainfiller/blob/main/brainfiller.snd.glsl)
#define repeatcenteredfloat(i, n) for(float i=-((n)-1.0)/2.0; i<=((n)-1.0)/2.0; i++)

layout(location = 0) uniform int waveOutPosition;
#if defined(EXPORT_EXECUTABLE)
vec2 resolution = {
        SCREEN_XRESO,
        SCREEN_YRESO
    };
#define NUM_SAMPLES_PER_SEC 48000.
float time = waveOutPosition / NUM_SAMPLES_PER_SEC;
#else
layout(location = 2) uniform float time;
layout(location = 3) uniform vec2 resolution;
#endif

out vec4 outColor;

const float BPM = 170;
const float BPS = BPM / 60.f;

const float SAMPLES_PER_SEC = 48000.; // 48000 in minimal_gl

const int SAMPLES_PER_STEP = int(SAMPLES_PER_SEC / BPS / 4.0);
const int SAMPLES_PER_BEAT = 4 * SAMPLES_PER_STEP;

const float T2B = BPS;
const float B2T = 1.0 / BPS;

const float PI = 3.14159265358979323846;
const float TAU = 2 * PI;

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

// NOTE: Shader Minifier ignores mat3(1.0), which should initialize a diagonal matrix
//       So we manually set the last entry and hope the uninitialized matrix is all zeroes
mat3 R(float x, int o) {
    vec2 v = sincos(x);
    int a = o, b = (o + 1) % 3, c = (o + 2) % 3;
    mat3 m = mat3(0.0);
    m[a][a] = v.x;
    m[b][a] = v.y;
    m[a][b] = -v.y;
    m[b][b] = v.x;
    m[c][c] = 1.0;
    return m;
}
/* R(t,0)=rotX, R(t,1)=rotY, R(t,2)=rotZ */

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

float sdBox(vec2 p, vec2 b)
{
    vec2 d = abs(p) - b;
    return length(max(d, 0.0))
        + min(max(d.x, d.y), 0.0); // remove this line for an only partially signed sdf
}

float sdBox(vec2 p, vec2 ll, vec2 ur)
{
    vec2 d = ur - ll;
    return sdBox(p - ll - d / 2.0, d / 2.0);
}

float sdRoundBox(vec2 p, vec2 b, float r)
{
    vec2 d = abs(p) - b;
    return length(max(d, 0.0)) - r
        + min(max(d.x, d.y), 0.0); // remove this line for an only partially signed sdf
}

float sdRoundBox(vec2 p, vec2 ll, vec2 ur, float r)
{
    vec2 d = ur - ll;
    return sdRoundBox(p - ll - d / 2.0, d / 2.0 - r, r);
}

// Slope -1 through a
float sdLine2(vec2 p, vec2 a) {
    return dot(p - a, vec2(1.0 / sqrt(2.0)));
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
    float c = sdRoundBox(p, G(0, 0), G(3, 6), rounding);
    d = max(d, c);

    return d;
}

float Q(vec2 p) {
    float d = sdBox(p, G(0, 0), G(3, 5));
    d = max(d, -sdBox(p, G(1, 1), G(2, 3)));

    // rounding
    float c = sdRoundBox(p, G(0, 0), G(3, 5), rounding);
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

// CREDIT: Hash functions from brainfiller/Ob5vr (https://github.com/0b5vr/brainfiller/blob/main/brainfiller.snd.glsl)
uvec3 hash3u(uvec3 v) {
    v = v * 1664525u + 1013904223u;

    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;

    v ^= v >> 16u;

    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;

    return v;
}

vec3 hash3f_normalized(vec3 v) {
    uvec3 u = hash3u(floatBitsToUint(v));
    return (vec3(u) / float(-1u)) * 2.0 - 1.0;
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

vec3 main_fnuque(vec2 uv, float time, float noise_mag, float noise_res, float noise_prob) {
    // vec3 p = vec3(0.);
    // p.z-=time;

    // vec3 d = normalize(vec3(uv, -1));

    //outColor = vec4(uv, 0, 1);
    // outColor = scene0(p, R(time*0.1, 0)*R(0.3, 1)*d, time);

    // uv.x += time*0.1;
    uv.y += 0.35;

    if (true) {
        vec2 uvd = floor(uv * (1.0 + noise_res * abs(sin(floor(time * 5.0) * 23.762))));
        vec3 offset = vec3(uvd, 0.1);
        vec3 noise = hash3f_normalized(offset + floor(time * 16.0) + vec3(10.897, 77.98, 0.112));

        if (noise.z > (1.0 - noise_prob)) {
            uv += (noise.xy) * noise_mag;
        }

        // outColor = vec4(hash3f_normalized(vec3(uvd, time*0.1)), 1);
        // return;
    }

    float d, q;

    vec2 p = uv * 20.0;
    p.x *= 1.0 + p.y / 55.0;
    p.x += 29.5;
    d = FNUQUE(p);
    q = smoothstep(0.0, -0.05, d);
    return vec3(q);
}

vec2 cam_shake(float time) {
    vec2 offset = vec2(0.0, 0.0);
    offset += sin(100.0 * time * vec2(1.2, 1.44) + vec2(1.32, 0.43)) * 0.1;
    offset += sin(100.0 * time * vec2(1.94, 1.56) + vec2(2.39, 5.18)) * 0.07;
    return offset * 0.5;
}

// CREDIT: Inspiration from https://www.shadertoy.com/view/3cScWy by Xor
// NOTE: Very sensitive to FOV, should be 1 to match above
//       Varying tunnel radius also interesting
//       Can be changed to sphere by including z in cylinder length
//       Coloring, multiplier on i is interesting
//       Instead of dividing with d on coloring line, can multiply
//       Multiplier on sin in offset can control randomness, going quite extreme
//       Coloring can be given an offset to desaturate a bit
//       Letting rotation depend on time can give more life
vec4 scene0(vec3 p_in, vec3 dir, float time, float sphereness, float noisyness, float exposure, float wildness, float rounding_multiplier) {
    vec3 p = p_in;
    vec4 O = vec4(0.);
    float d = 0, z = 0, i = 0;
    for (; i < 3e1; i++) {
        // Cylinder + Gyroid
        //		d=abs(length(p.xy)-5. + dot(sin(p.xyz), cos(p.yzx)));
        vec3 p_ = R(0.01, 1) * (p * 0.9);
        // vec3 p_ = p;
        // 0.003 is just the best constant
        float cyl_sphere_dist = mix(length(p.xy), length(p.xyz), sphereness);
        d = .003 + abs(cyl_sphere_dist - 4. + noisyness * dot(sin(p_), cos(p_).yzx));
        z += d;
        O += (1.0 + sin(i * 0.3 + z + time + vec4(6 * sin(time + z * 0.2 + 0.5), 1, 2 * sin(time * 0.1 + z * 0.9 + 0.3), 0))) / d;
        // O+=(1.+sin(i*0.3+z+time+vec4(6,1,2,0)))/d;
        p += dir * d;
        //p=z*dir;

        for (float f = 1.; f++ < 2.;
            //Blocky, stretched waves
            p += wildness * sin(round(p.yxz * rounding_multiplier) / 3. * f) / f);
    }
    // float c=length(p)*0.01;
    // return vec4(vec3(c), 1.);
    return tanh(O / exposure);
}

// CREDIT: Inspiration from https://www.shadertoy.com/view/WcKXDV by Xor
vec4 scene1(vec3 dir, float time, float sphereness, float planeness, float wildness, float rounding_multiplier, float camshake_timescale, float camshake_magnitude, float exposure)
{
    vec4 O = vec4(0.);
    //Raymarch depth
    float z,
    //Step distance
    d,
    //Raymarch iterator
    i;
    //Clear fragColor and raymarch 20 steps
    for (O *= i; i++ < 2e1; )
    {
        //Sample point (from ray direction)
        vec3 p = vec3(cam_shake(time * camshake_timescale) * camshake_magnitude, 0.0) + z * dir;

        //Polar coordinates and additional transformations
        // float sphereness = -0.1 + sin(time*0.1); // 0.0 = cylinder, 1.0 = sphere, small negative values are interesting!
        float cyl_sphere_plane_dist = mix(length(p.xy), length(p.xyz), sphereness);
        cyl_sphere_plane_dist = mix(cyl_sphere_plane_dist, length(p.x), planeness);
        // float cyl_sphere_dist = mix(length(p.xy), length(p.xyz), sphereness);
        p = vec3(atan(p.y / .2, p.x) * 2., p.z / 4., cyl_sphere_plane_dist - 4. - z * .5);
        //        p += 0.3*sin(p.zyx*3.);
        //           p = round(p.xzy*10.1)/10.1;

        //Apply turbulence and refraction effect
        for (d = 0.; d++ < 7.; )
            p += wildness * sin(round(p.yzx * rounding_multiplier) / 3. * d + 0.8 * time + 0.3 * cos(d)) / d;

        //Distance to cylinder and waves with refraction
        z += d = length(vec4(.4 * cos(p) - 0.4, p.z));

        //Coloring and brightness
        // O += (1.1+cos(p.x+i*.4+z+vec4(6,1,2,0)))/d;
        O += (1.0 + sin(i * 0.3 + z + time + vec4(6 * sin(time + z * 0.2 + 0.5), 1, 2 * sin(time * 0.1 + z * 0.9 + 0.3), 0))) / d;
    }
    //Tanh tonemap
    O = tanh(O * O / exposure);
    return O;
}

// CREDIT: Inspiration from https://www.shadertoy.com/view/3fBcR3
vec4 scene2(vec3 dir, float time, float rounding_multiplier) {
    // void mainImage(out vec4 o, vec2 u) {

    vec4 o = vec4(0.);

    vec3 q, p;

    float i, s,
    // start the ray at a small random distance,
    // this will reduce banding
    //   d = .125*texelFetch(iChannel0, ivec2(uv)%1024, 0).a,
    d = 0.0, // Rework this
    t = time;

    // scale coords
    //u =(u+u-p.xy)/p.y;

    for (o *= i; i++ < 1e2; ) {

        // shorthand for standard raymarch sample, then move forward:
        // p = ro + rd * d, p.z -= 5.;
        // q = p = vec3(u * d, d - 5.);
        q = p = dir * d + vec3(0., 0., 20.0 - time * 0.4);

        // turbulence
        for (s = 1.; s++ < 8.;
            q += sin(.6 * t + p.zxy * s * .3) * .4,
            p += sin(t + round(p.yzx * rounding_multiplier) / rounding_multiplier * s) * .25);

        // distance to spheres
        float dist_sphere = .005 + abs(-(length(p.xyz * 2.03 + 1.) - 2. - length(q.xyz - 1.) - 5.)) * .2;
        float dist_cyl = .005 + abs(-(length(p.xyz * 2.03 + 1.) - 2. - length(q.x - 1.) - 5.)) * .2;
        // d += s = mix(dist_sphere, dist_cyl, -0.1 + 1.4*sin(time*0.1));
        d += s = dist_cyl;

        // color: 1.+cos so we don't go negative, cos(d+vec4(6,4,2,0)) samples from the palette
        // divide by s for form and distance
        o += (1.0 + cos(p.z + vec4(3, 4, 2, 0))) / s;
    }

    // tonemap and divide brightness
    o = sqrt(tanh(o / 4e3));
    return o;
    // }
}

void main() {
    vec2 uv = (gl_FragCoord.xy * 2 - resolution) / resolution.yy;

    const int InitialQuietSamples = int(SAMPLES_PER_SEC * 0.5);
    if (waveOutPosition < InitialQuietSamples) {
        outColor = vec4(vec3(0.0), 1.0);
        return;
    }

    // vec4 time = vec4((waveOutPosition + samp_offset - InitialQuietSamples) % (SAMPLES_PER_BEAT * ivec4(1, 4, 64, 65536))) / SAMPLES_PER_SEC;
    vec4 music_time = vec4((waveOutPosition - InitialQuietSamples) % (SAMPLES_PER_BEAT * ivec4(1, 4, 64, 65536))) / SAMPLES_PER_SEC;
    vec4 step = music_time * BPS;
    vec4 beat = step / 4.0;

    vec3 e0_p = vec3(0.);
    e0_p.z -= music_time.w * 0.4;

    float e0_FOV = 3.0; // 1.0 Is nice initially, but for sphereness animation, higher is better probably
    float e0_x_rot = music_time.w * 0.1;
    float e0_look_rot = PI + 0.08; // Looking backwards in z is nice
    float e0_sphereness = -0.1 + sin(music_time.w * 0.1); // 0.0 = cylinder, 1.0 = sphere, small negative values are interesting!
    // OKAY, the above animation is MEGANICE
    // float sphereness = 0.0;
    // float noisyness = 10.0; // Values up to 10 look interesting
    // If we tweak multiplier in tanh filter, we loose outer edges/interesting behaviour
    float e0_noisyness = 1.0; // Values up to 10 look interesting, more animating
    // float exposure = 1e2;
    float e0_exposure = 1e2;
    float e0_wildness = 0.5;
    // float wildness = 1.5; // Also good
    // float wildness = 0.01; // Extremely low values of this are also nice!
    float e0_rounding_multiplier = 6.0; // 1->30 are interesting
    vec3 e0_d = normalize(vec3(uv, -e0_FOV));

    // Version 2
    // float e0_noisyness = 2.0; // Values up to 10 look interesting, more animating
    // float e0_exposure = 300.0;
    // float e0_wildness = 0.1;
    // float e0_rounding_multiplier = 6.0; // 1->30 are interesting
    // vec3 e0_d = normalize(vec3(uv, -e0_FOV));

    // Version 3
    // float e0_noisyness = 1.0; // Values up to 10 look interesting, more animating
    // float e0_exposure = 1e2;
    // float e0_wildness = 0.1;
    // float e0_rounding_multiplier = 20.0; // 1->30 are interesting
    // vec3 e0_d = normalize(vec3(uv, -e0_FOV));

    float s1_FOV = 0.5; // 0.5 is good, 1 2 also work;
    float s1_x_rot = time * 0.1;
    float s1_look_rot = 0.3; // PI*0.5 is also interesting
    float s1_sphereness = 0.2 + 1.4 * sin(time * 0.1);
    float s1_planeness = 0.5 + sin(time * 0.19);
    // float sphereness = 0.5;
    // float planenses = 3.5;
    float s1_wildness = 3.5;
    float s1_rounding_multiplier = 2.7; // 0.3, 0.7, 8 good, 192 good
    vec3 s1_d = normalize(vec3(uv, -s1_FOV));

    float s2_FOV = 0.2;
    float s2_x_rot = music_time.w * 0.1;
    float s2_look_rot = 0.6;
    float s2_rounding_multiplier = 1.0 / 1.0; // 1/8, 1, 8 works
    vec3 s2_d = normalize(vec3(uv, -s2_FOV));

    // Default scene 0
    if (false) {
        outColor = scene0(e0_p, R(e0_x_rot, 0) * R(e0_look_rot, 1) * e0_d, time, e0_sphereness, e0_noisyness, e0_exposure, e0_wildness, e0_rounding_multiplier);
    }

    // Default scene 1
    if (false) {
        float effect_time = time;
        outColor = scene1(R(s1_x_rot, 0) * R(s1_look_rot, 1) * s1_d, effect_time, s1_sphereness, s1_planeness, s1_wildness, s1_rounding_multiplier, 1.0, 1.0, 4e2);
    }

    // Default scene 2
    if (false) {
        outColor = scene2(R(s2_x_rot, 0) * R(s2_look_rot, 1) * s2_d, music_time.w, s2_rounding_multiplier);
    }

    if (true) {
        // INTRO
        if (step.w >= 0.0 && step.w < 63.0) {
            // float s2_FOV = 1.2 + 0.1 * hash3f_normalized(vec3(floor(music_time.w * 6.0))).x;
            float s2_FOV = 1.0; // + 0.05 * sin(music_time.w * 30.0) + 0.05 * sin(music_time.w * 27.0);
            vec3 s2_d = normalize(vec3(uv, -s2_FOV));
            float s2_look_rot = 1.4 - music_time.w * 0.05;
            float dissolve_time = max(0.0, step.w - 55.0);
            outColor = scene2(R(s2_x_rot, 0) * R(s2_look_rot, 1) * s2_d, music_time.w, s2_rounding_multiplier * 0.6 * pow(2.0, 0.10 * (pow(dissolve_time, 1.5))));
        }

        // Circle closing
        // if (step.w >= 56.0 && step.w < 64.0) {
        //     float effect_time = (music_time.w - 56.0 * B2T) * 2.0 - 4.5;
        //     float sphereness = -0.1 + sin(effect_time * 0.1); // 0.0 = cylinder, 1.0 = sphere, small negative values are interesting!

        //     vec3 p = vec3(0.);
        //     p.z -= effect_time * 0.5 * 0.4;

        //     float fade_in = pow(smoothstep(0.0, 8.0, step.w - 56.0), 4.0);

        //     outColor = fade_in * scene0(p, R(e0_x_rot, 0) * R(e0_look_rot, 1) * e0_d, effect_time, sphereness, e0_noisyness, e0_exposure, e0_wildness, e0_rounding_multiplier);
        // }

        // BODY BLOCK 1
        // TWO VARIANTS
        if (true) {
            if (step.w >= 64.0 && step.w < 128.0) {
                float effect_time = (music_time.w - 64.0 * B2T) * 1.5 + 33.0;
                float warp_time = effect_time * 1.2;
                float s1_sphereness = 0.2 + 1.4 * sin(warp_time * 0.1);
                float s1_planeness = 0.5 + sin(warp_time * 0.19);
                outColor = scene1(R(s1_x_rot, 0) * R(s1_look_rot, 1) * s1_d, effect_time * 0.4, s1_sphereness, s1_planeness, s1_wildness, s1_rounding_multiplier, 1.0, 1.0, 4e2);
            }
        } else {
            if (step.w >= 64.0 && step.w < 128.0) {
                float effect_time = (music_time.w - 64.0 * B2T) * 1.0 + 31.0;
                float s1_sphereness = 0.2 + 1.4 * sin(effect_time * 0.1);
                float s1_planeness = 0.5 + sin(effect_time * 0.19);
                outColor = scene1(R(s1_x_rot, 0) * R(s1_look_rot, 1) * s1_d, effect_time, s1_sphereness, s1_planeness, s1_wildness, s1_rounding_multiplier, 1.0, 1.0, 4e2);
            }
        }
        // BODY BLOCK 2
        if (step.w >= 128.0 && step.w < 160.0) {
            float effect_time = (music_time.w - 128.0 * B2T) * 0.9 + 28.0;
            float sphereness = -0.1 + sin(effect_time * 0.1); // 0.0 = cylinder, 1.0 = sphere, small negative values are interesting!

            vec3 p = vec3(0.);
            p.z -= effect_time * 0.4;

            outColor = scene0(p, R(e0_x_rot, 0) * R(e0_look_rot, 1) * e0_d, effect_time, sphereness, e0_noisyness, e0_exposure, e0_wildness, e0_rounding_multiplier);
        }
        if (step.w >= 160.0 && step.w < 192.0) {
            float effect_time = (music_time.w - 128.0 * B2T) * 0.73 + 50.0;
            float sphereness = -0.1 + sin(effect_time * 0.1); // 0.0 = cylinder, 1.0 = sphere, small negative values are interesting!

            vec3 p = vec3(0.);
            p.z -= effect_time * 0.4;

            outColor = scene0(p, R(e0_x_rot, 0) * R(e0_look_rot, 1) * e0_d, effect_time, sphereness, e0_noisyness, e0_exposure, e0_wildness, e0_rounding_multiplier);
        }
        // BREAK
        // if (step.w >= 192.0 && step.w < 256.0) {
        //     float effect_time = (music_time.w - 192.0 * B2T) * 1.0 + 2.0;
        //     e0_sphereness = -0.1 + 0.2 * sin(-0.2 + effect_time * 0.120); // 0.0 = cylinder, 1.0 = sphere, small negative values are interesting!

        //     // vec3 p = vec3(0.);
        //     // p.z -= effect_time * 0.5 * 0.4;

        //     outColor = scene0(-e0_p + 5.0 * 0.15 * (effect_time - 5.0), R(e0_x_rot + 1.8, 0) * R(e0_look_rot + 0.4 * effect_time / 20.0, 1) * e0_d, music_time.w * 0.1, e0_sphereness, e0_noisyness, e0_exposure * 2.5, e0_wildness, e0_rounding_multiplier);
        // }
        if (step.w >= 192.0 && step.w < 256.0) {
            float effect_time = (music_time.w - 192.0 * B2T) * 0.71 + 15.0;

            float s1_sphereness = 0.2 + 1.4 * sin(effect_time * 0.1);
            float s1_planeness = 0.5 + sin(effect_time * 0.19);
            float s1_look_rot = 1.3; // PI*0.5 is also interesting
            float s1_wildness = 0.1;
            float s1_rounding_multiplier = 0.7; // 0.3, 0.7, 8 good, 192 good

            // Run from around 7 seconds
            outColor = scene1(R(s1_x_rot, 0) * R(s1_look_rot + cam_shake(effect_time * 0.005).x * 10.0, 1) * s1_d, effect_time, s1_sphereness, s1_planeness, s1_wildness, s1_rounding_multiplier, 1.0, 1.0, 4e3);

            float q = (step.w - 192.0);
            float bar = floor(q / 4.0);
            float in_bar = mod(q, 4.0);

            if (in_bar >= 2.5 && step.w < 248.0) {
                float noise_mag = 0.2;
                float noise_res = 1.0;
                float noise_prob = 1.85;

                outColor.xyz += 1.2 * main_fnuque(uv + vec2(cam_shake(time).x, 0.0) * 10.0 * exp(-in_bar * 1.5), bar, noise_mag, noise_res, noise_prob);
            }
        }

        // if (step.w >= 224.0 && step.w < 256.0) {
        //     float effect_time = (music_time.w - 224.0 * B2T) * 0.5 + 29.7 + 5.65;

        //     float s2_rounding_multiplier = 1.0 / 16.0; // 1/8, 1, 8 works

        //     // Run from around 28-30 seconds
        //     outColor = scene2(R(s2_x_rot, 0) * R(s2_look_rot, 1) * s2_d, effect_time, s2_rounding_multiplier);
        // }
        // BODY BLOCK 3
        if (step.w >= 256.0 && step.w < 320.0) {
            float effect_time = (music_time.w - 256.0 * B2T) * 1.0 + 31.0;
            float s1_sphereness = 0.2 + 1.4 * sin(effect_time * 0.1);
            float s1_planeness = 0.5 + sin(effect_time * 0.19);
            outColor = scene1(R(s1_x_rot, 0) * R(s1_look_rot, 1) * s1_d, effect_time, s1_sphereness, s1_planeness, s1_wildness, s1_rounding_multiplier, 1.0, 1.0, 4e2);
        }
        // OUTTRO
        if (step.w >= 328.0) {
            float effect_time = (music_time.w - (256.0 + 64.0 + 8.0) * B2T) * 0.75;

            float noise_mag = 0.1;
            float noise_res = 2.5;
            float noise_prob = 0.25;

            vec3 col = vec3(0.0);
            col += main_fnuque(uv / (1.0 + effect_time * 0.01) + vec2(cam_shake(time).x, 0.0) * 0.05, effect_time, 0.0 * noise_mag, noise_res, noise_prob);
            // col += 0.5 * main_fnuque(uv * exp(0.05 * hash3f_normalized(vec3(floor(time * 10.0))).r), effect_time, noise_mag, noise_res, noise_prob);

            // const float N = 19.0;
            // repeatcenteredfloat(i, N)
            // {
            //     col.r += main_fnuque(uv * exp(-0.015 + i * 0.003), effect_time, noise_mag, noise_res, noise_prob).r;
            //     col.g += main_fnuque(uv * exp(0.000 + i * 0.003), effect_time, noise_mag, noise_res, noise_prob).g;
            //     col.b += main_fnuque(uv * exp(+0.015 + i * 0.003), effect_time, noise_mag, noise_res, noise_prob).b;
            // }
            // col /= N;

            outColor = vec4(col, 1.0);
        }
    }

    if (false) {
        vec3 col = vec3(0.0);

        float noise_mag = 0.3;
        float noise_res = 2.5;
        float noise_prob = 0.85;

        const float N = 19.0;
        repeatcenteredfloat(i, N)
        {
            col.r += main_fnuque(uv * exp(-0.015 + i * 0.003), time * 0.0, noise_mag, noise_res, noise_prob).r;
            col.g += main_fnuque(uv * exp(0.000 + i * 0.003), time * 0.0, noise_mag, noise_res, noise_prob).g;
            col.b += main_fnuque(uv * exp(+0.015 + i * 0.003), time * 0.0, noise_mag, noise_res, noise_prob).b;
        }
        col /= N;

        outColor = vec4(col, 1.0);
    }

    vec2 uv2 = (gl_FragCoord.xy * 2 - resolution) / resolution.xy;
    outColor.xyz += 0.03 * hash3f_normalized(vec3(uv2, music_time.w));
    outColor.xyz = sqrt(outColor.xyz);
    outColor.xyz *= clamp(1.5 - 1.1 * length(uv2), 0.05, 1.0);
}

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
