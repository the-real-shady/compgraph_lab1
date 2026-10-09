#version 450

layout(location = 0) in vec3 frag_color;
layout(location = 1) in vec3 frag_view_position;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform Uniforms {
	mat4 mvp;
	mat4 model_view;
	vec4 tint;
	float shading;
} uniforms;

void main() {
	if (uniforms.shading == 0.0) {
		out_color = vec4(frag_color * uniforms.tint.rgb, 1.0);
		return;
	}

	vec3 normal = normalize(cross(dFdx(frag_view_position), dFdy(frag_view_position)));

	if (normal.z < 0.0) {
		normal = -normal;
	}

	const vec3 light_direction = normalize(vec3(0.35, 0.6, 0.7));

	float diffuse = max(dot(normal, light_direction), 0.0);
	float lighting = 0.28 + 0.72 * diffuse;

	out_color = vec4(frag_color * uniforms.tint.rgb * lighting, 1.0);
}
