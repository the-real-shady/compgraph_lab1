#include "application.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <imgui.h>

#include <vulkan/vulkan.h>

#include "math.hpp"

namespace application {

namespace {

using graphics::internal::context;

struct Vertex {
	float position[3];
	float color[3];
};

struct Uniforms {
	math::Mat4 mvp;
	math::Mat4 model_view;
	float tint[4];
	float shading;
};

constexpr math::Vec3 vertex_positions[] = {
	{  1.0f,  0.0f,  0.0f },
	{ -1.0f,  0.0f,  0.0f },
	{  0.0f,  1.0f,  0.0f },
	{  0.0f, -1.0f,  0.0f },
	{  0.0f,  0.0f,  1.0f },
	{  0.0f,  0.0f, -1.0f },
};

constexpr uint16_t indices[] = {
	2, 4, 0,  2, 1, 4,  2, 5, 1,  2, 0, 5,
	3, 0, 4,  3, 4, 1,  3, 1, 5,  3, 5, 0,
};

constexpr uint32_t index_count = sizeof(indices) / sizeof(indices[0]);
constexpr uint32_t max_objects = 3;

struct Object {
	math::Vec3 position = { 0.0f, 0.0f, 0.0f };
	math::Vec3 rotation = { 0.0f, 0.0f, 0.0f };
	math::Vec3 scale = { 1.0f, 1.0f, 1.0f };
	math::Vec3 tint = { 1.0f, 1.0f, 1.0f };
	float orbit_phase = 0.0f;

	VkBuffer uniform_buffer = VK_NULL_HANDLE;
	VmaAllocation uniform_allocation = VK_NULL_HANDLE;
	void* uniform_mapped = nullptr;
	VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
};

Object objects[max_objects] = {
	{ .rotation = { 20.0f, 30.0f, 0.0f } },
	{ .scale = { 0.6f, 0.6f, 0.6f }, .tint = { 1.0f, 0.65f, 0.3f },
	  .orbit_phase = 2.0f * math::pi / 3.0f },
	{ .scale = { 0.45f, 0.45f, 0.45f }, .tint = { 0.4f, 0.8f, 1.0f },
	  .orbit_phase = 4.0f * math::pi / 3.0f },
};

int object_count = 3;
int selected_object = 0;

VkBuffer vk_vertex_buffer;
VmaAllocation vma_vertex_allocation;
VkBuffer vk_index_buffer;
VmaAllocation vma_index_allocation;

VkDescriptorSetLayout vk_descriptor_set_layout;
VkDescriptorPool vk_descriptor_pool;

VkPipelineLayout vk_pipeline_layout;
VkPipeline vk_pipeline_culling;
VkPipeline vk_pipeline_no_culling;

enum class Projection : int {
	perspective = 0,
	orthographic = 1,
};

enum class Trajectory : int {
	none = 0,
	circle = 1,
	lissajous = 2,
};

double current_time;
bool shading_enabled = true;
bool culling_enabled = true;

Projection projection_mode = Projection::perspective;
float field_of_view = 60.0f;
float orthographic_height = 5.0f;
float camera_distance = 6.0f;

Trajectory trajectory = Trajectory::circle;
bool animation_playing = true;
float animation_speed = 1.0f;
float trajectory_radius = 1.5f;
float lissajous_frequency_x = 3.0f;
float lissajous_frequency_y = 2.0f;
bool spin_enabled = true;
float spin_speed = 60.0f;

double animation_time;
double previous_time;
double spin_angle;

bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                  VkBuffer& buffer, VmaAllocation& allocation, void** mapped) {
	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	const VmaAllocationCreateInfo allocation_info = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
		         VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	VmaAllocationInfo result = {};
	if (vmaCreateBuffer(context.allocator, &buffer_info, &allocation_info,
	                    &buffer, &allocation, &result) != VK_SUCCESS) {
		std::cerr << "Failed to create buffer of " << size << " bytes\n";
		return false;
	}

	if (mapped != nullptr) {
		*mapped = result.pMappedData;
	}

	return true;
}

bool loadShaderModule(const char* path, VkShaderModule& module) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file) {
		std::cerr << "Failed to open shader " << path << '\n';
		return false;
	}

	const std::streamsize size = file.tellg();
	std::vector<uint32_t> code(size_t(size) / sizeof(uint32_t));
	file.seekg(0);
	file.read(reinterpret_cast<char*>(code.data()), size);

	const VkShaderModuleCreateInfo module_info = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size_t(size),
		.pCode = code.data(),
	};

	if (vkCreateShaderModule(context.device, &module_info, nullptr, &module) != VK_SUCCESS) {
		std::cerr << "Failed to create shader module for " << path << '\n';
		return false;
	}

	return true;
}

bool createPipelines() {
	VkShaderModule vertex_shader = VK_NULL_HANDLE;
	VkShaderModule fragment_shader = VK_NULL_HANDLE;

	if (!loadShaderModule("shaders/octahedron.vert.spv", vertex_shader) ||
	    !loadShaderModule("shaders/octahedron.frag.spv", fragment_shader)) {
		return false;
	}

	const VkPipelineShaderStageCreateInfo stages[] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_shader,
			.pName = "main",
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_shader,
			.pName = "main",
		},
	};

	const VkVertexInputBindingDescription binding = {
		.binding = 0,
		.stride = sizeof(Vertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};

	const VkVertexInputAttributeDescription attributes[] = {
		{ .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, position) },
		{ .location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, color) },
	};

	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &binding,
		.vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
		.pVertexAttributeDescriptions = attributes,
	};

	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
	};

	const VkPipelineViewportStateCreateInfo viewport_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};

	const VkDynamicState dynamic_states[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};

	const VkPipelineDynamicStateCreateInfo dynamic_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = sizeof(dynamic_states) / sizeof(dynamic_states[0]),
		.pDynamicStates = dynamic_states,
	};

	VkPipelineRasterizationStateCreateInfo rasterization = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_BACK_BIT,
		.frontFace = VK_FRONT_FACE_CLOCKWISE,
		.lineWidth = 1.0f,
	};

	const VkPipelineMultisampleStateCreateInfo multisample = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};

	const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS,
	};

	const VkPipelineColorBlendAttachmentState color_blend_attachment = {
		.blendEnable = VK_FALSE,
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	};

	const VkPipelineColorBlendStateCreateInfo color_blend = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &color_blend_attachment,
	};

	const VkPipelineLayoutCreateInfo pipeline_layout = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &vk_descriptor_set_layout,
	};

	if (vkCreatePipelineLayout(context.device, &pipeline_layout, nullptr,
	                           &vk_pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create pipeline layout\n";
		return false;
	}

	VkGraphicsPipelineCreateInfo pipeline = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = sizeof(stages) / sizeof(stages[0]),
		.pStages = stages,
		.pVertexInputState = &vertex_input,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport_state,
		.pRasterizationState = &rasterization,
		.pMultisampleState = &multisample,
		.pDepthStencilState = &depth_stencil,
		.pColorBlendState = &color_blend,
		.pDynamicState = &dynamic_state,
		.layout = vk_pipeline_layout,
		.renderPass = context.render_pass,
		.subpass = 0,
	};

	VkResult result = vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1,
	                                            &pipeline, nullptr, &vk_pipeline_culling);

	if (result == VK_SUCCESS) {
		rasterization.cullMode = VK_CULL_MODE_NONE;
		result = vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1,
		                                   &pipeline, nullptr, &vk_pipeline_no_culling);
	}

	vkDestroyShaderModule(context.device, fragment_shader, nullptr);
	vkDestroyShaderModule(context.device, vertex_shader, nullptr);

	if (result != VK_SUCCESS) {
		std::cerr << "Failed to create graphics pipeline\n";
		return false;
	}

	return true;
}

bool createDescriptors() {
	const VkDescriptorSetLayoutBinding binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
	};

	const VkDescriptorSetLayoutCreateInfo layout = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &binding,
	};

	if (vkCreateDescriptorSetLayout(context.device, &layout, nullptr,
	                                &vk_descriptor_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor set layout\n";
		return false;
	}

	const VkDescriptorPoolSize pool_size = {
		.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = max_objects,
	};

	const VkDescriptorPoolCreateInfo pool = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = max_objects,
		.poolSizeCount = 1,
		.pPoolSizes = &pool_size,
	};

	if (vkCreateDescriptorPool(context.device, &pool, nullptr, &vk_descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor pool\n";
		return false;
	}

	VkDescriptorSetLayout layouts[max_objects];
	VkDescriptorSet sets[max_objects];
	for (uint32_t i = 0; i < max_objects; ++i) {
		layouts[i] = vk_descriptor_set_layout;
	}

	const VkDescriptorSetAllocateInfo allocate = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = vk_descriptor_pool,
		.descriptorSetCount = max_objects,
		.pSetLayouts = layouts,
	};

	if (vkAllocateDescriptorSets(context.device, &allocate, sets) != VK_SUCCESS) {
		std::cerr << "Failed to allocate descriptor sets\n";
		return false;
	}

	VkDescriptorBufferInfo buffer_infos[max_objects];
	VkWriteDescriptorSet writes[max_objects];

	for (uint32_t i = 0; i < max_objects; ++i) {
		objects[i].descriptor_set = sets[i];

		buffer_infos[i] = {
			.buffer = objects[i].uniform_buffer,
			.offset = 0,
			.range = sizeof(Uniforms),
		};

		writes[i] = {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = sets[i],
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &buffer_infos[i],
		};
	}

	vkUpdateDescriptorSets(context.device, max_objects, writes, 0, nullptr);

	return true;
}

math::Vec3 trajectoryOffset(float phase) {
	const float t = float(animation_time) + phase;

	switch (trajectory) {
	case Trajectory::circle:
		return { trajectory_radius * std::cos(t), 0.0f, trajectory_radius * std::sin(t) };

	case Trajectory::lissajous:
		return { trajectory_radius * std::sin(lissajous_frequency_x * t),
		         trajectory_radius * std::sin(lissajous_frequency_y * t + math::pi * 0.5f),
		         0.0f };

	case Trajectory::none:
		break;
	}

	return { 0.0f, 0.0f, 0.0f };
}

}

bool initialize() {
	Vertex vertices[sizeof(vertex_positions) / sizeof(vertex_positions[0])] = {};
	for (size_t i = 0; i < sizeof(vertices) / sizeof(vertices[0]); ++i) {
		const math::Vec3 p = vertex_positions[i];
		const math::Vec3 c = math::normalize(p) * 0.5f + math::Vec3{ 0.5f, 0.5f, 0.5f };

		vertices[i] = { { p.x, p.y, p.z }, { c.x, c.y, c.z } };
	}

	void* vertex_mapped = nullptr;
	if (!createBuffer(sizeof(vertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
	                  vk_vertex_buffer, vma_vertex_allocation, &vertex_mapped)) {
		return false;
	}
	std::memcpy(vertex_mapped, vertices, sizeof(vertices));

	void* index_mapped = nullptr;
	if (!createBuffer(sizeof(indices), VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
	                  vk_index_buffer, vma_index_allocation, &index_mapped)) {
		return false;
	}
	std::memcpy(index_mapped, indices, sizeof(indices));

	for (uint32_t i = 0; i < max_objects; ++i) {
		if (!createBuffer(sizeof(Uniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		                  objects[i].uniform_buffer, objects[i].uniform_allocation,
		                  &objects[i].uniform_mapped)) {
			return false;
		}
	}

	if (!createDescriptors() || !createPipelines()) {
		return false;
	}

	return true;
}

void shutdown() {
	vkQueueWaitIdle(context.graphics_queue);

	vkDestroyPipeline(context.device, vk_pipeline_no_culling, nullptr);
	vkDestroyPipeline(context.device, vk_pipeline_culling, nullptr);
	vkDestroyPipelineLayout(context.device, vk_pipeline_layout, nullptr);

	vkDestroyDescriptorPool(context.device, vk_descriptor_pool, nullptr);
	vkDestroyDescriptorSetLayout(context.device, vk_descriptor_set_layout, nullptr);

	for (uint32_t i = 0; i < max_objects; ++i) {
		vmaDestroyBuffer(context.allocator, objects[i].uniform_buffer,
		                 objects[i].uniform_allocation);
	}

	vmaDestroyBuffer(context.allocator, vk_index_buffer, vma_index_allocation);
	vmaDestroyBuffer(context.allocator, vk_vertex_buffer, vma_vertex_allocation);
}

void update(double time) {
	const double delta = previous_time > 0.0 ? time - previous_time : 0.0;
	previous_time = time;
	current_time = time;

	if (animation_playing) {
		animation_time += delta * double(animation_speed);

		if (spin_enabled) {
			spin_angle += delta * double(animation_speed) * double(spin_speed);
		}
	}

	ImGui::Begin("Lab 1: octahedron");
	ImGui::Text("%.1f FPS", double(ImGui::GetIO().Framerate));

	if (ImGui::CollapsingHeader("Projection", ImGuiTreeNodeFlags_DefaultOpen)) {
		int mode = int(projection_mode);
		ImGui::RadioButton("Perspective", &mode, int(Projection::perspective));
		ImGui::SameLine();
		ImGui::RadioButton("Orthographic", &mode, int(Projection::orthographic));
		projection_mode = Projection(mode);

		if (projection_mode == Projection::perspective) {
			ImGui::SliderFloat("Field of view", &field_of_view, 20.0f, 110.0f, "%.0f deg");
		} else {
			ImGui::SliderFloat("Visible height", &orthographic_height, 1.0f, 15.0f);
		}

		ImGui::SliderFloat("Camera distance", &camera_distance, 2.0f, 20.0f);
	}

	if (ImGui::CollapsingHeader("Objects", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::SliderInt("Count", &object_count, 1, int(max_objects));

		if (selected_object >= object_count) {
			selected_object = object_count - 1;
		}

		ImGui::Combo("Selected", &selected_object, "Object 1\0Object 2\0Object 3\0");

		Object& object = objects[selected_object];

		ImGui::SliderFloat3("Position", &object.position.x, -3.0f, 3.0f);
		ImGui::SliderFloat3("Rotation", &object.rotation.x, -180.0f, 180.0f, "%.0f deg");
		ImGui::SliderFloat3("Scale", &object.scale.x, 0.1f, 3.0f);
		ImGui::ColorEdit3("Tint", &object.tint.x);
		ImGui::SliderFloat("Orbit phase", &object.orbit_phase, 0.0f, 2.0f * math::pi, "%.2f rad");

		if (ImGui::Button("Reset object")) {
			object.position = { 0.0f, 0.0f, 0.0f };
			object.rotation = { 0.0f, 0.0f, 0.0f };
			object.scale = { 1.0f, 1.0f, 1.0f };
			object.tint = { 1.0f, 1.0f, 1.0f };
		}
	}

	if (ImGui::CollapsingHeader("Animation", ImGuiTreeNodeFlags_DefaultOpen)) {
		if (ImGui::Button(animation_playing ? "Pause" : "Play")) {
			animation_playing = !animation_playing;
		}

		ImGui::SameLine();
		if (ImGui::Button("Restart")) {
			animation_time = 0.0;
			spin_angle = 0.0;
		}

		ImGui::SliderFloat("Speed", &animation_speed, 0.0f, 3.0f);

		int selected = int(trajectory);
		ImGui::Combo("Trajectory", &selected, "None\0Circle\0Lissajous\0");
		trajectory = Trajectory(selected);

		if (trajectory != Trajectory::none) {
			ImGui::SliderFloat("Radius", &trajectory_radius, 0.0f, 3.0f);
		}

		if (trajectory == Trajectory::lissajous) {
			ImGui::SliderFloat("Frequency X", &lissajous_frequency_x, 1.0f, 6.0f, "%.1f");
			ImGui::SliderFloat("Frequency Y", &lissajous_frequency_y, 1.0f, 6.0f, "%.1f");
		}

		ImGui::Checkbox("Spin", &spin_enabled);
		if (spin_enabled) {
			ImGui::SliderFloat("Spin speed", &spin_speed, 0.0f, 240.0f, "%.0f deg/s");
		}
	}

	if (ImGui::CollapsingHeader("Rendering", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::Checkbox("Flat shading", &shading_enabled);
		ImGui::Checkbox("Back-face culling", &culling_enabled);
	}

	ImGui::End();
}

void render(const graphics::internal::FrameData& fd) {
	const float aspect = float(context.swapchain_extent.width) /
	                     float(context.swapchain_extent.height);

	const math::Mat4 view = math::lookAt({ 0.0f, 0.0f, camera_distance },
	                                     { 0.0f, 0.0f, 0.0f },
	                                     { 0.0f, 1.0f, 0.0f });

	math::Mat4 projection;
	if (projection_mode == Projection::perspective) {
		projection = math::perspective(math::radians(field_of_view), aspect, 0.1f, 100.0f);
	} else {
		const float half_height = orthographic_height * 0.5f;
		const float half_width = half_height * aspect;

		projection = math::ortho(-half_width, half_width, -half_height, half_height, 0.1f, 100.0f);
	}

	for (int i = 0; i < object_count; ++i) {
		const Object& object = objects[i];

		const math::Mat4 model =
			math::translate(object.position + trajectoryOffset(object.orbit_phase)) *
			math::rotate({ math::radians(object.rotation.x),
			               math::radians(object.rotation.y + float(spin_angle)),
			               math::radians(object.rotation.z) }) *
			math::scale(object.scale);

		const math::Mat4 model_view = view * model;

		const Uniforms uniforms = {
			projection * model_view,
			model_view,
			{ object.tint.x, object.tint.y, object.tint.z, 1.0f },
			shading_enabled ? 1.0f : 0.0f,
		};

		std::memcpy(object.uniform_mapped, &uniforms, sizeof(uniforms));
	}

	vkResetCommandBuffer(fd.command_buffer, 0);

	const VkCommandBufferBeginInfo command_buffer_begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};

	vkBeginCommandBuffer(fd.command_buffer, &command_buffer_begin);

	const VkClearValue clear_values[] = {
		{ .color = { { 0.05f, 0.06f, 0.09f, 1.0f } } },
		{ .depthStencil = { 1.0f, 0 } },
	};

	const VkRenderPassBeginInfo render_pass_begin = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = context.render_pass,
		.framebuffer = fd.framebuffer,
		.renderArea = { .extent = context.swapchain_extent },
		.clearValueCount = sizeof(clear_values) / sizeof(clear_values[0]),
		.pClearValues = clear_values,
	};

	vkCmdBeginRenderPass(fd.command_buffer, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);

	const VkViewport viewport = {
		.width = float(context.swapchain_extent.width),
		.height = float(context.swapchain_extent.height),
		.maxDepth = 1.0f,
	};

	const VkRect2D scissor = { .extent = context.swapchain_extent };

	vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
	vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

	vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
	                  culling_enabled ? vk_pipeline_culling : vk_pipeline_no_culling);

	const VkDeviceSize offset = 0;
	vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vk_vertex_buffer, &offset);
	vkCmdBindIndexBuffer(fd.command_buffer, vk_index_buffer, 0, VK_INDEX_TYPE_UINT16);

	for (int i = 0; i < object_count; ++i) {
		vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
		                        vk_pipeline_layout, 0, 1, &objects[i].descriptor_set,
		                        0, nullptr);

		vkCmdDrawIndexed(fd.command_buffer, index_count, 1, 0, 0, 0);
	}

	vkCmdEndRenderPass(fd.command_buffer);

	vkEndCommandBuffer(fd.command_buffer);
}

}
