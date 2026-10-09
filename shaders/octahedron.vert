#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;

layout(location = 0) out vec3 frag_color;
layout(location = 1) out vec3 frag_view_position;

layout(set = 0, binding = 0) uniform Uniforms {
	mat4 mvp;
	mat4 model_view;
	vec4 tint;
	float shading;
} uniforms;

void main() {
	gl_Position = uniforms.mvp * vec4(in_position, 1.0);

	frag_color = in_color;
	frag_view_position = (uniforms.model_view * vec4(in_position, 1.0)).xyz;
}
