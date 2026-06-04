#version 450

layout(location = 0) in vec3 vColor;
layout(location = 0) out vec4 outColor;

void main() {
    // round point sprites: discard fragments outside unit circle
    vec2 d = gl_PointCoord - vec2(0.5);
    if (dot(d, d) > 0.25) discard;
    outColor = vec4(vColor, 1.0);
}
