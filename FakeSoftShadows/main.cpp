#if defined(_DEBUG)
#define DEBUG
#endif

#define BOOST_SCOPE_EXIT_CONFIG_USE_LAMBDAS

#define STEP1_SHADOWMAP
#define STEP2_SMOOTHIES
#define STEP3_ALPHA_CORRECTION

#include <stdexcept>
#include <set>
#include <array>
#include <vector>
#include <memory>
#include <numeric>
#include <list>
#include <cassert>
#include <sstream>
#include <iostream>
#include <tuple>
#include <map>
#include <cmath>
#include <chrono>
#include <thread>
#include <mutex>
#include <future>

#include <SDL.h>
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/norm.hpp>
#include <nana/gui.hpp>
#include <nana/gui/widgets/group.hpp>
#include <nana/gui/widgets/label.hpp>
#include <nana/gui/widgets/spinbox.hpp>

#include <boost/optional.hpp>
#include <boost/scope_exit.hpp>

#define STRINGIFY_(x) #x
#define STRINGIFY(x) STRINGIFY_(x)
#define THROW_ERROR() { std::ostringstream oss; oss << __FILE__ ":" STRINGIFY(__LINE__) << ": Error"; throw std::runtime_error(oss.str()); }
#define THROW_SDL_ERROR() { std::ostringstream oss; oss << __FILE__ ":" STRINGIFY(__LINE__) << ": " << SDL_GetError() << std::endl; throw std::runtime_error(oss.str()); }

#if defined(DEBUG)
#define XGL_POST() { \
	if(glGetError() != GL_NO_ERROR) { \
		assert(!"OpenGL call failed"); \
	} \
}
#else
#define XGL_POST()
#endif
#define XGL(EXPR) { EXPR; XGL_POST() }

static const auto window_width(640);
static const auto window_height(480);
// bias constant is inside the shader

enum {
	debug_none,
	debug_shadowmap,
	debug_smoothie_depth,
	debug_smoothie_alpha
} debug_mode = debug_none;
static auto smoothie_size(0.08f);
static auto sm_width(1024);
static auto sm_height(1024);

static float light_vertices[] = {0.0f, 0.0f, 0.0f};

static float tex_debug_vertices[] = {
	-1.0f, 1.0f, 0.0f, 1.0f,
	-1.0f, -1.0f, 0.0f, 0.0f,
	1.0f, 1.0f, 1.0f, 1.0f,
	-1.0f, -1.0f, 0.0f, 0.0f,
	1.0f, -1.0f, 1.0f, 0.0f,
	1.0f, 1.0f, 1.0f, 1.0f
};

static float cube_vertices[] =
#include "models/cube_vertices.inc"
;
static float cube_normals[] =
#include "models/cube_normals.inc"
;
static unsigned int cube_indices[] =
#include "models/cube_indices.inc"
;

static float sphere_vertices[] =
#include "models/sphere_vertices.inc"
;
static float sphere_normals[] =
#include "models/sphere_normals.inc"
;
static unsigned int sphere_indices[] =
#include "models/sphere_indices.inc"
;

static float cylinder_vertices[] =
#include "models/cylinder_vertices.inc"
;
static float cylinder_normals[] =
#include "models/cylinder_normals.inc"
;
static unsigned int cylinder_indices[] =
#include "models/cylinder_indices.inc"
;

static float cone_vertices[] =
#include "models/cone_vertices.inc"
;
static float cone_normals[] =
#include "models/cone_normals.inc"
;
static unsigned int cone_indices[] = 
#include "models/cone_indices.inc"
;

static float floor_vertices[] = 
#include "models/floor_vertices.inc"
;
static float floor_normals[] =
#include "models/floor_normals.inc"
;
static unsigned int floor_indices[] =
#include "models/floor_indices.inc"
;

static float teapot_vertices[] = 
#include "models/teapot_vertices.inc"
;
static float teapot_normals[] =
#include "models/teapot_normals.inc"
;
static unsigned int teapot_indices[] =
#include "models/teapot_indices.inc"
;

static float monkey_vertices[] = 
#include "models/monkey_vertices.inc"
;
static float monkey_normals[] =
#include "models/monkey_normals.inc"
;
static unsigned int monkey_indices[] =
#include "models/monkey_indices.inc"
;

static float fence_vertices[] = 
#include "models/fence_vertices.inc"
;
static float fence_normals[] =
#include "models/fence_normals.inc"
;
static unsigned int fence_indices[] =
#include "models/fence_indices.inc"
;


static const char *tex_debug_vert_src = R"GLSL(
#version 140
in vec2 pos;
in vec2 uv;
out vec2 v_uv;

void main() {
	v_uv = uv;
	gl_Position = vec4(pos, 0.0, 1.0);
}
)GLSL";

static const char *tex_debug_frag_src = R"GLSL(
#version 140
uniform int mode;
uniform sampler2D tex;
in vec2 v_uv;
out vec4 f_color;

void main() {
	if(mode == 0) { // depth
		float val = 1.0 - texture(tex, v_uv).r/10.0;
		f_color = vec4(val, val, val, 1.0);
	} else { // alpha
		float val = texture(tex, v_uv).r;
		f_color = vec4(val, val, val, 1.0);
	}
}
)GLSL";

static const char *sm_object_vert_src = R"GLSL(
#version 140

uniform mat4 proj;
uniform mat4 view;
uniform mat4 model_rotate;
uniform mat4 model_translate;
in vec3 pos;
in vec3 normal;
out float v_depth;

void main() {
	vec4 world_pos = view * model_translate * model_rotate * vec4(pos, 1.0);
	v_depth = -world_pos.z;
	gl_Position = proj * world_pos; 
}
)GLSL";

static const char *sm_object_frag_src = R"GLSL(
#version 140

in float v_depth;
out vec4 f_depth;

void main() {
	f_depth = vec4(v_depth);
}
)GLSL";

static const char *smoothie_edges_vert_src = R"GLSL(
#version 140

in vec2 pos;
in float depth;
in float alpha;
out vec2 v_pos;
out float v_depth;
out float v_alpha;

void main() {
	v_pos = pos;
	v_depth = depth;
	v_alpha = alpha;
	gl_Position = vec4(pos, 0.0, 1.0);
}
)GLSL";

static const char *smoothie_edges_frag_src = R"GLSL(
#version 140

uniform sampler2D sm;
in vec2 v_pos;
in float v_depth;
in float v_alpha;
out vec4 f_depth;
out vec4 f_alpha;

void main() {
	float r = texture(sm, (v_pos+1.0)/2.0).r;
	float factor = )GLSL" 
#if defined(STEP3_ALPHA_CORRECTION)
	"1.0 - v_depth/r;"
#else 
	"1.0;"
#endif
	R"GLSL(
	float a_prime = factor < 0.001 ? 1.0 : clamp(v_alpha/factor, 0.0, 1.0);
	f_depth = vec4(v_depth);
	f_alpha = vec4(a_prime);
}
)GLSL";

static const char *smoothie_corners_vert_src = R"GLSL(
#version 140

in vec2 pos;
in float depth;
in vec2 origin;
out vec2 v_pos;
out float v_depth;
out vec2 v_origin;

void main() {
	v_pos = pos;
	v_depth = depth;
	v_origin = origin;
	gl_Position = vec4(pos, 0.0, 1.0);
}
)GLSL";

static const char *smoothie_corners_frag_src = R"GLSL(
#version 140

uniform float smoothie_size;
uniform sampler2D sm;
in vec2 v_pos;
in float v_depth;
in vec2 v_origin;
out vec4 f_depth;
out vec4 f_alpha;

void main() {
	float r = texture(sm, (v_pos+1.0)/2.0).r;
	float a = clamp(length(v_pos - v_origin)/smoothie_size, 0.0, 1.0);
	float factor = )GLSL"
#if defined(STEP3_ALPHA_CORRECTION)
	"1.0 - v_depth/r;"
#else
	"1.0;"
#endif
	R"GLSL(
	float a_prime = factor < 0.001 ? 1.0 : clamp(a/factor, 0.0, 1.0);
	f_depth = vec4(v_depth);
	f_alpha = vec4(a_prime);
}
)GLSL";

static const char *cam_object_vert_src = R"GLSL(
#version 140

uniform mat4 proj;
uniform mat4 view;
uniform mat4 model_rotate;
uniform mat4 model_translate;
in vec3 pos;
in vec3 normal;
out vec3 v_world_pos;
out vec3 v_world_normal;

void main() {
	vec4 world_pos = model_translate * model_rotate * vec4(pos, 1.0);
	v_world_pos = world_pos.xyz;
	v_world_normal = (model_rotate * vec4(normal, 1.0)).xyz;
	gl_Position = proj * view * world_pos;
}
)GLSL";

static const char *cam_object_frag_src = R"GLSL(
#version 140

uniform vec3 light_world_pos;
uniform sampler2D sm;
uniform sampler2D smoothie_depth;
uniform sampler2D smoothie_alpha;
uniform mat4 sm_proj;
uniform mat4 sm_view;
in vec3 v_world_pos;
in vec3 v_world_normal;
out vec4 f_color;

void main() {
	float bias = -0.15;
	float ambient = 0.1;
	float diffuse = max(0.0, dot(normalize(light_world_pos - v_world_pos), v_world_normal));
	vec4 sm_world_pos = sm_view * vec4(v_world_pos, 1.0);
	vec4 sm_pos = sm_proj * sm_world_pos;
	float depth = -sm_world_pos.z + bias;
	vec2 sm_uv = ((sm_pos.xy/sm_pos.w + 1.0)/2.0) * textureSize(sm, 0);
	ivec2 sm_uv_i = ivec2(sm_uv);
	float xf = fract(sm_uv.x);
	float yf = fract(sm_uv.y);
	ivec2 xr = xf >= 0.5 ? ivec2(0, 1) : ivec2(0, -1);
	ivec2 yr = yf >= 0.5 ? ivec2(0, 1) : ivec2(0, -1);
	vec2 xw = vec2(1.0 - abs(xf - 0.5), abs(xf - 0.5));
	vec2 yw = vec2(1.0 - abs(yf - 0.5), abs(yf - 0.5));
	float shadow = 0.0f;
	for(int j = 0; j != 2; ++j) {
		float row_shadow = 0.0f;
		for(int i = 0; i != 2; ++i) {
			ivec2 uv_i = sm_uv_i + ivec2(xr[i], yr[j]);
			float sm_d = texelFetch(sm, uv_i, 0).r;
			float smoothie_d = texelFetch(smoothie_depth, uv_i, 0).r;
			float sample_shadow = depth > sm_d ? 0.0f : depth > smoothie_d ? texelFetch(smoothie_alpha, uv_i, 0).r : 1.0f;
			row_shadow += xw[i]*sample_shadow;
		}
		shadow += yw[j]*row_shadow;
	}
	f_color = clamp(ambient + shadow * diffuse, 0.0, 1.0) * vec4(1.0);
}
)GLSL";

static const char *cam_object_basic_vert_src = R"GLSL(
#version 140

uniform mat4 proj;
uniform mat4 view;
uniform mat4 model_rotate;
uniform mat4 model_translate;
in vec3 pos;
in vec3 normal;
out vec3 v_world_pos;
out vec3 v_world_normal;

void main() {
	vec4 world_pos = model_translate * model_rotate * vec4(pos, 1.0);
	v_world_pos = world_pos.xyz;
	v_world_normal = (model_rotate * vec4(normal, 1.0)).xyz;
	gl_Position = proj * view * world_pos;
}
)GLSL";

static const char *cam_object_basic_frag_src = R"GLSL(
#version 140

uniform vec3 light_world_pos;
uniform sampler2D sm;
uniform mat4 sm_proj;
uniform mat4 sm_view;
in vec3 v_world_pos;
in vec3 v_world_normal;
out vec4 f_color;

void main() {
	float bias = -0.15;
	float ambient = 0.1;
	float diffuse = max(0.0, dot(normalize(light_world_pos - v_world_pos), v_world_normal));
	vec4 sm_world_pos = sm_view * vec4(v_world_pos, 1.0);
	vec4 sm_pos = sm_proj * sm_world_pos;
	float depth = -sm_world_pos.z + bias;
	float sm_d = texture(sm, (sm_pos.xy/sm_pos.w+1.0)/2.0).r;
	float shadow = depth > sm_d ? 0.0f : 1.0f;
	f_color = clamp(ambient + shadow * diffuse, 0.0, 1.0) * vec4(1.0);
}
)GLSL";

static const char *cam_light_vert_src = R"GLSL(
#version 140

uniform mat4 proj;
uniform mat4 view;
uniform mat4 model;
in vec3 pos;

void main() {
	gl_PointSize = 6;
	gl_Position = proj * view * model * vec4(pos, 1.0);
}
)GLSL";

static const char *cam_light_frag_src = R"GLSL(
#version 140

out vec4 f_color;

void main() {
	vec2 p = 2*(gl_PointCoord.st - 0.5);
	float mask = p.x*p.x + p.y*p.y <= 1.0 ? 1.0 : 0.0;
	f_color = mask * vec4(1.0);
}
)GLSL";

// OBJ typically has vertices defined with the Y axis being up, and negative Z-axis being forward: use this rotation to transform into our space
static const glm::mat4 obj_rot(glm::rotate(glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f)));

static glm::vec2 intersect_lines(const glm::vec2 &o0, const glm::vec2 &n0, const glm::vec2 &o1, const glm::vec2 &n1) {
	return {
		(-n0.y*(n1.x*o1.x + n1.y*o1.y) + n1.y*(n0.x*o0.x + n0.y*o0.y))/(n0.x*n1.y - n0.y*n1.x),
		(n0.x*(n1.x*o1.x + n1.y*o1.y) - n1.x*(n0.x*o0.x + n0.y*o0.y))/(n0.x*n1.y - n0.y*n1.x)
	};
}

template <typename Vec>
static glm::vec3 xyz(const Vec &v) {
	return {v.x, v.y, v.z};
}

template <typename Vec>
static glm::vec2 xy(const Vec &v) {
	return {v.x, v.y};
}

template <typename Vec>
static glm::vec2 yx(const Vec &v) {
	return {v.y, v.x};
}

struct program {
	program() {
		vert_shader = glCreateShader(GL_VERTEX_SHADER);
		XGL_POST();
		frag_shader = glCreateShader(GL_FRAGMENT_SHADER);
		XGL_POST();
		prog = glCreateProgram();
		XGL_POST();
	}

	program(const program &) = delete;

	~program() {
		XGL(glDeleteProgram(prog));
		XGL(glDeleteShader(frag_shader));
		XGL(glDeleteShader(vert_shader));
	}

	static std::unique_ptr<program> compile(const char *vert_src, const char *frag_src, const std::map<std::string, int> &frag_data_bindings) {
		std::unique_ptr<program> prog(new program());
		XGL(glShaderSource(prog->vert_shader, 1, &vert_src, nullptr));
		XGL(glShaderSource(prog->frag_shader, 1, &frag_src, nullptr));
		XGL(glCompileShader(prog->vert_shader));
		GLint res;
		XGL(glGetShaderiv(prog->vert_shader, GL_COMPILE_STATUS, &res));
		if(res == GL_FALSE) {
			char buf[1024];
			XGL(glGetShaderInfoLog(prog->vert_shader, (sizeof buf)-1, nullptr, buf));
			std::ostringstream oss;
			oss << "failed to compile vertex shader:\n" << buf;
			throw std::runtime_error(oss.str());
		}
		XGL(glCompileShader(prog->frag_shader));
		XGL(glGetShaderiv(prog->frag_shader, GL_COMPILE_STATUS, &res));
		if(res == GL_FALSE) {
			char buf[1024];
			XGL(glGetShaderInfoLog(prog->frag_shader, (sizeof buf)-1, nullptr, buf));
			std::ostringstream oss;
			oss << "failed to compile fragment shader:\n" << buf;
			throw std::runtime_error(oss.str());
		}
		XGL(glAttachShader(prog->prog, prog->vert_shader));
		XGL(glAttachShader(prog->prog, prog->frag_shader));
		for(const auto &p : frag_data_bindings) {
			XGL(glBindFragDataLocation(prog->prog, p.second, p.first.c_str()));
		}
		XGL(glLinkProgram(prog->prog));
		XGL(glGetProgramiv(prog->prog, GL_LINK_STATUS, &res));
		if(res == GL_FALSE) {
			char buf[1024];
			XGL(glGetProgramInfoLog(prog->prog, (sizeof buf)-1, nullptr, buf));
			std::ostringstream oss;
			oss << "failed to link program:\n" << buf;
			throw std::runtime_error(oss.str());
		}
		return prog;
	}
	
	GLuint vert_shader;
	GLuint frag_shader;
	GLuint prog;
};

struct shared {
	std::unique_ptr<program> tex_debug_prog;
	std::unique_ptr<program> sm_object_prog;
	std::unique_ptr<program> smoothie_edges_prog;
	std::unique_ptr<program> smoothie_corners_prog;
	std::unique_ptr<program> cam_object_prog;
	std::unique_ptr<program> cam_object_basic_prog;
	std::unique_ptr<program> cam_light_prog;

	shared() :
		tex_debug_prog(program::compile(tex_debug_vert_src, tex_debug_frag_src, {{"f_color", 0}})),
		sm_object_prog(program::compile(sm_object_vert_src, sm_object_frag_src, {{"f_depth", 0}})),
		smoothie_edges_prog(program::compile(smoothie_edges_vert_src, smoothie_edges_frag_src, {{"f_depth", 0}, {"f_alpha", 1}})),
		smoothie_corners_prog(program::compile(smoothie_corners_vert_src, smoothie_corners_frag_src, {{"f_depth", 0}, {"f_alpha", 1}})),
		cam_light_prog(program::compile(cam_light_vert_src, cam_light_frag_src, {{"f_color", 0}})),
		cam_object_prog(program::compile(cam_object_vert_src, cam_object_frag_src, {{"f_color", 0}})),
		cam_object_basic_prog(program::compile(cam_object_basic_vert_src, cam_object_basic_frag_src, {{"f_color", 0}}))
	{}
};

struct light {
	light(const glm::vec3 &pos, const glm::vec3 &dir, const shared &s) :
		pos(pos),
		dir(dir)
	{
		XGL(glGenBuffers(1, &vbo));
		XGL(glBindBuffer(GL_ARRAY_BUFFER, vbo));
		XGL(glBufferData(GL_ARRAY_BUFFER, sizeof light_vertices, light_vertices, GL_STATIC_DRAW));

		XGL(glGenVertexArrays(1, &vao));
		XGL(glBindVertexArray(vao));
		{
			auto prog(s.cam_light_prog->prog);
			auto pos_loc(glGetAttribLocation(prog, "pos")); XGL_POST();
			XGL(glEnableVertexAttribArray(pos_loc));
			XGL(glVertexAttribPointer(pos_loc, 3, GL_FLOAT, GL_FALSE, sizeof(float)*3, reinterpret_cast<GLvoid *>(0)));
		}
	}

	light(const light &l) = delete;

	~light() {
		XGL(glDeleteBuffers(1, &vbo));
		XGL(glDeleteBuffers(1, &vao));
	}

	glm::mat4 proj(int width, int height) const {
		return glm::perspective(glm::radians(120.0f), float(width)/height, 1.0f, 20.0f);
	}

	glm::mat4 view() const {
		return glm::lookAt(pos, pos + dir, glm::vec3(0.0f, 0.0f, 1.0f));
	}

	GLuint vao;
	GLuint vbo;
	glm::vec3 pos;
	glm::vec3 dir;
};

struct object {
	object(float *vertices, size_t vertices_len, float *normals, size_t normals_len, unsigned int *indices, size_t indices_len, const glm::vec3 &pos, const glm::mat4 &rot, const shared &s) :
		pos(pos),
		rot(rot),
		vertices(vertices),
		vertices_len(vertices_len),
		normals(normals),
		normals_len(normals_len),
		indices(indices),
		indices_len(indices_len)
	{
		XGL(glGenBuffers(1, &vbo));
		XGL(glBindBuffer(GL_ARRAY_BUFFER, vbo));
		XGL(glBufferData(GL_ARRAY_BUFFER, vertices_len*(sizeof vertices[0]), vertices, GL_STATIC_DRAW));

		XGL(glGenBuffers(1, &nbo));
		XGL(glBindBuffer(GL_ARRAY_BUFFER, nbo));
		XGL(glBufferData(GL_ARRAY_BUFFER, normals_len*(sizeof normals[0]), normals, GL_STATIC_DRAW));

		XGL(glGenVertexArrays(1, &vao));
		XGL(glBindVertexArray(vao));
		{
			auto prog(s.cam_object_prog->prog);
			auto pos_loc(glGetAttribLocation(prog, "pos"));
			auto normal_loc(glGetAttribLocation(prog, "normal"));
			XGL(glEnableVertexAttribArray(pos_loc));
			XGL(glEnableVertexAttribArray(normal_loc));
			XGL(glBindBuffer(GL_ARRAY_BUFFER, vbo));
			XGL(glVertexAttribPointer(pos_loc, 3, GL_FLOAT, GL_FALSE, 0, reinterpret_cast<GLvoid *>(0)));
			XGL(glBindBuffer(GL_ARRAY_BUFFER, nbo));
			XGL(glVertexAttribPointer(normal_loc, 3, GL_FLOAT, GL_FALSE, 0, reinterpret_cast<GLvoid *>(0)));
		}
		XGL(glBindVertexArray(0));

		XGL(glGenBuffers(1, &ibo));
		XGL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo));
		XGL(glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_len*(sizeof indices[0]), indices, GL_STATIC_DRAW));
	}

	template <size_t VerticesLen, size_t NormalsLen, size_t IndicesLen>
	object(float(&vertices)[VerticesLen], float (&normals)[NormalsLen], unsigned int (&indices)[IndicesLen], const glm::vec3 &pos, const glm::mat4 &rot, const shared &s) :
		object(vertices, VerticesLen, normals, NormalsLen, indices, IndicesLen, pos, rot, s)
	{}

	object(const object &) = delete;

	~object() {
		XGL(glDeleteBuffers(1, &ibo));
		XGL(glDeleteBuffers(1, &nbo));
		XGL(glDeleteBuffers(1, &vbo));
		XGL(glDeleteVertexArrays(1, &vao));
	}

	glm::mat4 model_translate() const {
		return glm::translate(pos);
	}

	GLuint vao;
	GLuint vbo;
	GLuint nbo;
	GLuint ibo;
	glm::vec3 pos;
	glm::mat4 rot;

	float *vertices;
	size_t vertices_len;
	float *normals;
	size_t normals_len;
	unsigned int *indices;
	size_t indices_len;
};

struct smoothie_buffer {
private:
	// lexicographic comparator for vec3
	struct vec3_compare {
		bool operator()(const glm::vec3 &a, const glm::vec3 &b) const {
			return a.x < b.x || (a.x == b.x && (a.y < b.y || (a.y == b.y && a.z < b.z)));
		}
	};

	struct edge : std::array<glm::vec3, 2> {
		edge() {}

		edge(const glm::vec3 &v0, const glm::vec3 &v1) :
			array{v0, v1}
		{}

		const glm::vec3 &v0() const {
			return (*this)[0];
		}

		glm::vec3 &v0() {
			return (*this)[0];
		}

		const glm::vec3 &v1() const {
			return (*this)[1];
		}

		glm::vec3 &v1() {
			return (*this)[1];
		}
	};

	// lexicographic comparator for edge
	struct edge_compare {
		bool operator()(const edge &a, const edge &b) const {
			return v3c(a.v0(), b.v0()) || (a.v0() == b.v0() && v3c(a.v1(), b.v1()));
		}

		vec3_compare v3c;
	};

	struct triangle : std::array<glm::vec3, 3> {
		triangle(const glm::vec3 &v0, const glm::vec3 &v1, const glm::vec3 &v2) :
			array{v0, v1, v2}
		{}

		const glm::vec3 &v0() const {
			return (*this)[0];
		}

		const glm::vec3 &v1() const {
			return (*this)[1];
		}
		
		const glm::vec3 &v2() const {
			return (*this)[2];
		}

		glm::vec3 &v0() {
			return (*this)[0];
		}

		glm::vec3 &v1() {
			return (*this)[1];
		}

		glm::vec3 &v2() {
			return (*this)[2];
		}

		edge e0() const {
			return {v0(), v1()};
		}

		edge e1() const {
			return {v1(), v2()};
		}

		edge e2() const {
			return {v0(), v2()};
		}

		std::array<edge, 3> edges() const {
			return {e0(), e1(), e2()};
		}

		glm::vec3 midpoint;
		glm::vec3 normal;
	};

	struct sil_edge {
		std::map<edge, std::vector<triangle>>::iterator it; 
		size_t visible_index; // the index of the visible triangle
	};

	struct smoothie_edge {
		glm::vec3 v0_c; // first edge vertex in camera space
		glm::vec3 v1_c; // second edge vertex in camera space
		glm::vec2 v0_s; // first edge vertex in screen space
		glm::vec2 v1_s; // second edge vertex in screen space
		glm::vec2 d; // the normalized direction of the smoothie edge in screen space
	};

public:
	smoothie_buffer(const light &l, const object &o, const shared &s) : 
		l(l),
		o(o),
		num_edge_vertices(0),
		num_corner_vertices(0)
	{
		assert(o.indices_len % 3 == 0);
		assert(o.vertices_len % 3 == 0);
		assert(o.normals_len % 3 == 0);

		// construct a mapping from edges to triangles
		for(auto i(0); i != o.indices_len; i += 3) {
			auto i0(o.indices[i]);
			auto i1(o.indices[i+1]);
			auto i2(o.indices[i+2]);
			triangle t(
				{o.vertices[i0*3], o.vertices[i0*3+1], o.vertices[i0*3+2]}, 
				{o.vertices[i1*3], o.vertices[i1*3+1], o.vertices[i1*3+2]},
				{o.vertices[i2*3], o.vertices[i2*3+1], o.vertices[i2*3+2]});
			std::sort(t.begin(), t.end(), vec3_compare());
			t.midpoint = (t.v0() + t.v1() + t.v2())/3.0f;
			glm::vec3 n0(o.normals[i0*3], o.normals[i0*3+1], o.normals[i0*3+2]);
			glm::vec3 n1(o.normals[i1*3], o.normals[i1*3+1], o.normals[i1*3+2]);
			glm::vec3 n2(o.normals[i2*3], o.normals[i2*3+1], o.normals[i2*3+2]);
			t.normal = glm::normalize(n0 + n1 + n2);
			for(auto &&e : t.edges()) {
				edges[e].push_back(t);
			}
		}

		// remove all edges not connected to two adjacent triangles
		for(auto it(edges.begin()); it != edges.end();) {
			if(it->second.size() != 2) {
				it = edges.erase(it);
			} else {
				++it;
			}
		}

		XGL(glGenVertexArrays(1, &edges_vao));
		XGL(glGenBuffers(1, &edges_vbo));
		XGL(glGenVertexArrays(1, &corners_vao));
		XGL(glGenBuffers(1, &corners_vbo));

		XGL(glBindVertexArray(edges_vao));
		{
			auto prog(s.smoothie_edges_prog->prog);
			auto pos_loc(glGetAttribLocation(prog, "pos")); XGL_POST();
			auto depth_loc(glGetAttribLocation(prog, "depth")); XGL_POST();
			auto alpha_loc(glGetAttribLocation(prog, "alpha")); XGL_POST();
			XGL(glEnableVertexAttribArray(pos_loc));
			XGL(glEnableVertexAttribArray(depth_loc));
			XGL(glEnableVertexAttribArray(alpha_loc));
			XGL(glBindBuffer(GL_ARRAY_BUFFER, edges_vbo));
			XGL(glVertexAttribPointer(pos_loc, 2, GL_FLOAT, GL_FALSE, sizeof(float)*4, reinterpret_cast<GLvoid *>(0)));
			XGL(glVertexAttribPointer(depth_loc, 1, GL_FLOAT, GL_FALSE, sizeof(float)*4, reinterpret_cast<GLvoid *>(sizeof(float)*2)));
			XGL(glVertexAttribPointer(alpha_loc, 1, GL_FLOAT, GL_FALSE, sizeof(float)*4, reinterpret_cast<GLvoid *>(sizeof(float)*3)));
		}

		XGL(glBindVertexArray(corners_vao));
		{
			auto prog(s.smoothie_corners_prog->prog);
			auto pos_loc(glGetAttribLocation(prog, "pos")); XGL_POST();
			auto depth_loc(glGetAttribLocation(prog, "depth")); XGL_POST();
			auto origin_loc(glGetAttribLocation(prog, "origin")); XGL_POST();
			XGL(glEnableVertexAttribArray(pos_loc));
			XGL(glEnableVertexAttribArray(depth_loc));
			XGL(glEnableVertexAttribArray(origin_loc));
			XGL(glBindBuffer(GL_ARRAY_BUFFER, corners_vbo));
			XGL(glVertexAttribPointer(pos_loc, 2, GL_FLOAT, GL_FALSE, sizeof(float)*5, reinterpret_cast<GLvoid *>(0)));
			XGL(glVertexAttribPointer(depth_loc, 1, GL_FLOAT, GL_FALSE, sizeof(float)*5, reinterpret_cast<GLvoid *>(sizeof(float)*2)));
			XGL(glVertexAttribPointer(origin_loc, 2, GL_FLOAT, GL_FALSE, sizeof(float)*5, reinterpret_cast<GLvoid *>(sizeof(float)*3)));
		}

		XGL(glBindVertexArray(0));
	}

	smoothie_buffer(const smoothie_buffer &) = delete;

	~smoothie_buffer() {
		XGL(glDeleteBuffers(1, &corners_vbo));
		XGL(glDeleteVertexArrays(1, &corners_vao));
		XGL(glDeleteBuffers(1, &edges_vbo));
		XGL(glDeleteVertexArrays(1, &edges_vao));
	}

	void build() {
		auto proj(l.proj(sm_width, sm_height));
		auto view(l.view());
		const auto &model_rotate(o.rot);
		auto model(o.model_translate() * model_rotate);
		sil_edges_buf.clear();
		sil_vertices_buf.clear();
		for(auto it(edges.begin()); it != edges.end(); ++it) {
			auto &tris(it->second);
			auto t0_away(glm::dot(l.pos - xyz(model * glm::vec4(tris[0].midpoint, 1.0f)), xyz(model_rotate * glm::vec4(tris[0].normal, 1.0f))) < 0);
			auto t1_away(glm::dot(l.pos - xyz(model * glm::vec4(tris[1].midpoint, 1.0f)), xyz(model_rotate * glm::vec4(tris[1].normal, 1.0f))) < 0);
			if(t0_away && !t1_away || !t0_away && t1_away) {
				sil_edge info;
				info.it = it;
				info.visible_index = t1_away ? 0 : 1;
				sil_edges_buf.push_back(info);
			}
		}
		for(auto &sil_e : sil_edges_buf) {
			for(const auto &v : sil_e.it->first) {
				auto &arr(sil_vertices_buf[v]);
				if(!arr[0]) {
					arr[0] = sil_e;
				} else if(!arr[1]) {
					arr[1] = sil_e;
				} 
			}
		}
		// remove all vertices that are not connected to two triangles
		for(auto it(sil_vertices_buf.begin()); it != sil_vertices_buf.end();) {
			if(!it->second[1]) {
				it = sil_vertices_buf.erase(it);
			} else {
				++it;
			}
		}
		// note: from this point we assume that each silhouette vertex has two adjacent silhouette edges (if this is not the case, there is a bug and boost::optional will provide us with an error during runtime)
		auto compute_smoothie_edge = [&](const sil_edge &sil_e) {
			smoothie_edge se;
			auto &e(sil_e.it->first);
			auto n(sil_e.it->second[0].normal);
			auto away_dir_w(xyz(model_rotate * glm::vec4(glm::normalize(sil_e.it->second[0].normal + sil_e.it->second[1].normal), 1.0f)));
			auto vM_w(xyz(model * glm::vec4((e.v0() + e.v1())/2.0f, 1.0f)));
			auto vM_p_w(vM_w + 0.00001f*away_dir_w); // use a very small offset to avoid escaping the clipping plane
			se.v0_c = xyz(view * model * glm::vec4(e.v0(), 1.0f));
			se.v1_c = xyz(view * model * glm::vec4(e.v1(), 1.0f));
			auto vM_s4(proj * view * glm::vec4(vM_w, 1.0f));
			auto vM_p_s4(proj * view * glm::vec4(vM_p_w, 1.0f));
			auto away_dir_s(xy(vM_p_s4)/vM_p_s4.w - xy(vM_s4)/vM_s4.w);
			auto v0_s4(proj * glm::vec4(se.v0_c, 1.0f));
			auto v1_s4(proj * glm::vec4(se.v1_c, 1.0f));
			se.v0_s = xy(v0_s4)/v0_s4.w;
			se.v1_s = xy(v1_s4)/v1_s4.w;
			se.d = glm::normalize((yx(se.v1_s) - yx(se.v0_s)) * glm::vec2(-1.0f, 1.0f));
			if(glm::dot(se.d, away_dir_s) < 0.0f) {
				se.d *= -1.0f;
			}
			return se;
		};
		// construct smoothie edges
		edge_data_buf.clear();
		for(auto &sil_e : sil_edges_buf) {
			auto se(compute_smoothie_edge(sil_e));
			// v0
			edge_data_buf.push_back(se.v0_s.x);
			edge_data_buf.push_back(se.v0_s.y);
			edge_data_buf.push_back(-se.v0_c.z);
			edge_data_buf.push_back(0.0f);
			// v0 + d
			edge_data_buf.push_back(se.v0_s.x + se.d.x * smoothie_size);
			edge_data_buf.push_back(se.v0_s.y + se.d.y * smoothie_size);

			edge_data_buf.push_back(-se.v0_c.z);
			edge_data_buf.push_back(1.0f);
			// v1 + d
			edge_data_buf.push_back(se.v1_s.x + se.d.x * smoothie_size);
			edge_data_buf.push_back(se.v1_s.y + se.d.y * smoothie_size);
			edge_data_buf.push_back(-se.v1_c.z);
			edge_data_buf.push_back(1.0f);
			// v1 + d
			edge_data_buf.push_back(se.v1_s.x + se.d.x * smoothie_size);
			edge_data_buf.push_back(se.v1_s.y + se.d.y * smoothie_size);
			edge_data_buf.push_back(-se.v1_c.z);
			edge_data_buf.push_back(1.0f);
			// v1
			edge_data_buf.push_back(se.v1_s.x);
			edge_data_buf.push_back(se.v1_s.y);
			edge_data_buf.push_back(-se.v1_c.z);
			edge_data_buf.push_back(0.0f);
			// v0
			edge_data_buf.push_back(se.v0_s.x);
			edge_data_buf.push_back(se.v0_s.y);
			edge_data_buf.push_back(-se.v0_c.z);
			edge_data_buf.push_back(0.0f);
		}
		XGL(glBindBuffer(GL_ARRAY_BUFFER, edges_vbo));
		XGL(glBufferData(GL_ARRAY_BUFFER, edge_data_buf.size()*(sizeof edge_data_buf[0]), edge_data_buf.data(), GL_STATIC_DRAW));
		num_edge_vertices = edge_data_buf.size()/4;
		// construct smoothie corners
		corner_data_buf.clear();
		for(auto &p : sil_vertices_buf) {
			const auto &v(p.first);
			auto v4_c(view * model * glm::vec4(v, 1.0f));
			auto v4_s(proj * v4_c);
			auto v_s(xy(v4_s)/v4_s.w);
			auto se0(compute_smoothie_edge(*p.second[0]));
			auto se1(compute_smoothie_edge(*p.second[1]));
			auto vA_s(v_s + se0.d * smoothie_size);
			auto vB_s(v_s + se1.d * smoothie_size);
			if(glm::distance2(vA_s, vB_s) < std::numeric_limits<float>::epsilon()) {
				continue;
			}
			auto vM_s(intersect_lines(vA_s, se0.d, vB_s, se1.d));
			// v
			corner_data_buf.push_back(v_s.x);
			corner_data_buf.push_back(v_s.y);
			corner_data_buf.push_back(-v4_c.z);
			corner_data_buf.push_back(v_s.x);
			corner_data_buf.push_back(v_s.y);
			// vA
			corner_data_buf.push_back(vA_s.x);
			corner_data_buf.push_back(vA_s.y);
			corner_data_buf.push_back(-v4_c.z);
			corner_data_buf.push_back(v_s.x);
			corner_data_buf.push_back(v_s.y);
			// vM
			corner_data_buf.push_back(vM_s.x);
			corner_data_buf.push_back(vM_s.y);
			corner_data_buf.push_back(-v4_c.z);
			corner_data_buf.push_back(v_s.x);
			corner_data_buf.push_back(v_s.y);
			// vM
			corner_data_buf.push_back(vM_s.x);
			corner_data_buf.push_back(vM_s.y);
			corner_data_buf.push_back(-v4_c.z);
			corner_data_buf.push_back(v_s.x);
			corner_data_buf.push_back(v_s.y);
			// vB
			corner_data_buf.push_back(vB_s.x);
			corner_data_buf.push_back(vB_s.y);
			corner_data_buf.push_back(-v4_c.z);
			corner_data_buf.push_back(v_s.x);
			corner_data_buf.push_back(v_s.y);
			// v
			corner_data_buf.push_back(v_s.x);
			corner_data_buf.push_back(v_s.y);
			corner_data_buf.push_back(-v4_c.z);
			corner_data_buf.push_back(v_s.x);
			corner_data_buf.push_back(v_s.y);
		}
		XGL(glBindBuffer(GL_ARRAY_BUFFER, corners_vbo));
		XGL(glBufferData(GL_ARRAY_BUFFER, corner_data_buf.size()*(sizeof corner_data_buf[0]), corner_data_buf.data(), GL_STATIC_DRAW));
		num_corner_vertices = corner_data_buf.size()/5;
	}

	const light &l;
	const object &o;
	std::map<edge, std::vector<triangle>, edge_compare> edges;

	std::vector<sil_edge> sil_edges_buf;
	std::map<glm::vec3, std::array<boost::optional<sil_edge &>, 2>, vec3_compare> sil_vertices_buf;
	std::vector<float> edge_data_buf; // [x, y, depth, alpha]
	std::vector<float> corner_data_buf; // [x, y, depth, origin_x, origin_y]

	GLuint edges_vao;
	GLuint edges_vbo;
	size_t num_edge_vertices;

	GLuint corners_vao;
	GLuint corners_vbo;
	size_t num_corner_vertices;
};

struct scene {
	void prepare(const shared &s) {
		assert(l_smoothie_buffers.size() == 0);
		for(auto &o : objects) {
			l_smoothie_buffers.emplace_back(new smoothie_buffer(*l, *o, s));
		}
		orig_light_pos = l->pos;
	}

	void reset() {
		l->pos = orig_light_pos;
	}

	std::unique_ptr<light> l;
	std::vector<std::unique_ptr<object>> objects;
	std::vector<std::unique_ptr<smoothie_buffer>> l_smoothie_buffers;
	glm::vec3 cam_pos;
	float cam_theta;
	float cam_phi;

	// computed by prepare()
	glm::vec3 orig_light_pos;
};

int main(int argc, char **argv) {
	using clock = std::chrono::high_resolution_clock;
	boost::optional<std::thread> cp_thread;
	boost::optional<nana::form> cp;
	std::recursive_mutex main_mutex;
	bool error = false;
	try {
		if(SDL_Init(SDL_INIT_VIDEO) < 0) {
			THROW_SDL_ERROR();
		}
		BOOST_SCOPE_EXIT(void) {
			SDL_Quit();
		} BOOST_SCOPE_EXIT_END

		auto window(SDL_CreateWindow("Render", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, window_width, window_height, SDL_WINDOW_OPENGL));
		if(!window) {
			THROW_SDL_ERROR();
		}
		BOOST_SCOPE_EXIT(window) {
			SDL_DestroyWindow(window);
		} BOOST_SCOPE_EXIT_END
		SDL_SetWindowResizable(window, SDL_FALSE);

		if(SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3) < 0) {
			THROW_SDL_ERROR();
		}
		if(SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1) < 0) {
			THROW_SDL_ERROR();
		}
		if(SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE) < 0) {
			THROW_SDL_ERROR();
		}

		auto context(SDL_GL_CreateContext(window));
        if(!context) {
            THROW_SDL_ERROR();
        }
        BOOST_SCOPE_EXIT(context) {
            SDL_GL_DeleteContext(context);
        } BOOST_SCOPE_EXIT_END

        if(SDL_GL_MakeCurrent(window, context) < 0) {
            THROW_SDL_ERROR();
        }

		if(glewInit() != GLEW_OK) {
			THROW_ERROR();
		}

		XGL(glEnable(GL_PROGRAM_POINT_SIZE));
		XGL(glEnable(GL_POINT_SPRITE));

		int width, height;
        SDL_GetWindowSize(window, &width, &height);

		shared s;

		GLuint screen_vbo;
		XGL(glGenBuffers(1, &screen_vbo));
		BOOST_SCOPE_EXIT(screen_vbo) {
			XGL(glDeleteBuffers(1, &screen_vbo));
		} BOOST_SCOPE_EXIT_END
		XGL(glBindBuffer(GL_ARRAY_BUFFER, screen_vbo));
		XGL(glBufferData(GL_ARRAY_BUFFER, sizeof tex_debug_vertices, tex_debug_vertices, GL_STATIC_DRAW));

		GLuint screen_vao;
		XGL(glGenVertexArrays(1, &screen_vao));
		BOOST_SCOPE_EXIT(screen_vao) {
			XGL(glDeleteVertexArrays(1, &screen_vao));
		} BOOST_SCOPE_EXIT_END
		XGL(glBindVertexArray(screen_vao));
		{
			auto prog(s.tex_debug_prog->prog);
			auto pos_loc(glGetAttribLocation(prog, "pos")); XGL_POST();
			auto uv_loc(glGetAttribLocation(prog, "uv")); XGL_POST();
			XGL(glEnableVertexAttribArray(pos_loc));
			XGL(glEnableVertexAttribArray(uv_loc));
			XGL(glVertexAttribPointer(pos_loc, 2, GL_FLOAT, GL_FALSE, sizeof(float)*4, reinterpret_cast<GLvoid *>(0)));
			XGL(glVertexAttribPointer(uv_loc, 2, GL_FLOAT, GL_FALSE, sizeof(float)*4, reinterpret_cast<GLvoid *>(sizeof(float)*2)));
		}
		XGL(glBindVertexArray(0));

		GLuint sm_fb;
		XGL(glGenFramebuffers(1, &sm_fb));
		BOOST_SCOPE_EXIT(sm_fb) {
			XGL(glDeleteFramebuffers(1, &sm_fb));
		} BOOST_SCOPE_EXIT_END

		GLuint sm_tex;
		XGL(glGenTextures(1, &sm_tex));
		BOOST_SCOPE_EXIT(sm_tex) {
			XGL(glDeleteTextures(1, &sm_tex));
		} BOOST_SCOPE_EXIT_END
		XGL(glActiveTexture(GL_TEXTURE0));
		XGL(glBindTexture(GL_TEXTURE_2D, sm_tex));
		XGL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
		XGL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
		XGL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
		XGL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
		XGL(glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, sm_width, sm_height, 0,	GL_RED, GL_FLOAT, nullptr));

		XGL(glBindFramebuffer(GL_FRAMEBUFFER, sm_fb));
		XGL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, sm_tex, 0));

		GLuint smoothie_fb;
		XGL(glGenFramebuffers(1, &smoothie_fb));
		BOOST_SCOPE_EXIT(smoothie_fb) {
			XGL(glDeleteFramebuffers(1, &smoothie_fb));
		} BOOST_SCOPE_EXIT_END

		GLuint smoothie_depth_tex;
		XGL(glGenTextures(1, &smoothie_depth_tex));
		BOOST_SCOPE_EXIT(smoothie_depth_tex) {
			XGL(glDeleteTextures(1, &smoothie_depth_tex));
		} BOOST_SCOPE_EXIT_END
		XGL(glActiveTexture(GL_TEXTURE0));
		XGL(glBindTexture(GL_TEXTURE_2D, smoothie_depth_tex));
		XGL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
		XGL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
		XGL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
		XGL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
		XGL(glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, sm_width, sm_height, 0, GL_RED, GL_FLOAT, nullptr));

		GLuint smoothie_alpha_tex;
		XGL(glGenTextures(1, &smoothie_alpha_tex));
		BOOST_SCOPE_EXIT(smoothie_alpha_tex) {
			XGL(glDeleteTextures(1, &smoothie_alpha_tex));
		} BOOST_SCOPE_EXIT_END
		XGL(glActiveTexture(GL_TEXTURE0));
		XGL(glBindTexture(GL_TEXTURE_2D, smoothie_alpha_tex));
		XGL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
		XGL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
		XGL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
		XGL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
		XGL(glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, sm_width, sm_height, 0, GL_RED, GL_FLOAT, nullptr));

		XGL(glBindFramebuffer(GL_FRAMEBUFFER, smoothie_fb));
		XGL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, smoothie_depth_tex, 0));
		XGL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, smoothie_alpha_tex, 0));

		std::vector<std::string> scene_order;
		std::map<std::string, std::unique_ptr<scene>> scenes;
		// Basic
		{
			std::unique_ptr<scene> scn(new scene());
			scn->l = std::unique_ptr<light>(new light({0.0f, 0.0f, 6.0f}, glm::normalize(glm::vec3(0.0f, 0.01f, -1.0f)), s));
			scn->objects.emplace_back(new object(floor_vertices, floor_normals, floor_indices, {0.0f, 0.0f, 0.0f}, obj_rot, s));
			// scn->objects.emplace_back(new object(cube_vertices, cube_normals, cube_indices, {-3.0f, 4.0f, 0.0f}, obj_rot, s));
			// scn->objects.emplace_back(new object(cube_vertices, cube_normals, cube_indices, {-3.0f, -3.0f, 0.0f}, obj_rot, s));
			scn->objects.emplace_back(new object(cylinder_vertices, cylinder_normals, cylinder_indices, {3.0f, -3.0f, 0.0f}, obj_rot, s));
			// scn->objects.emplace_back(new object(sphere_vertices, sphere_normals, sphere_indices, {3.0f, 5.0f, 0.0f}, obj_rot, s));
			scn->cam_pos = {13.5463f, -5.2634f, 12.7381f};
			scn->cam_theta = 160.0f;
			scn->cam_phi = 131.9f;
			scn->prepare(s);
			scene_order.push_back("Basic");
			scenes.emplace("Basic", std::move(scn));
		}
		// Overlap
		{
			std::unique_ptr<scene> scn(new scene());
			scn->l = std::unique_ptr<light>(new light({0.0f, 0.0f, 8.0f}, glm::normalize(glm::vec3(0.0f, 0.01f, -1.0f)), s));
			scn->objects.emplace_back(new object(floor_vertices, floor_normals, floor_indices, {0.0f, 0.0f, 0.0f}, obj_rot, s));
			scn->objects.emplace_back(new object(cube_vertices, cube_normals, cube_indices, {2.0f, 0.0f, 0.0f}, obj_rot, s));
			scn->objects.emplace_back(new object(cube_vertices, cube_normals, cube_indices, {3.0f, 1.0f, 2.5f}, obj_rot, s));
			scn->cam_pos = {-5.18233f, 6.67561f, 11.4894f};
			scn->cam_theta = 326.0f;
			scn->cam_phi = 126.0f;
			scn->prepare(s);
			scene_order.push_back("Overlap");
			scenes.emplace("Overlap", std::move(scn));
		}
		// Teapot
		{
			std::unique_ptr<scene> scn(new scene());
			scn->l = std::unique_ptr<light>(new light({0.0f, 8.0f, 7.0f}, glm::normalize(glm::vec3(0.0f, -1.0f, -1.0f)), s));
			scn->objects.emplace_back(new object(floor_vertices, floor_normals, floor_indices, {0.0f, 0.0f, 0.0f}, obj_rot, s));
			scn->objects.emplace_back(new object(teapot_vertices, teapot_normals, teapot_indices, {0.0f, 3.0f, 0.0f}, obj_rot, s));
			scn->cam_pos = {-6.11014f,12.9f,9.67555f};
			scn->cam_theta = 301.0f;
			scn->cam_phi = 121.9f;
			scn->prepare(s);
			scene_order.push_back("Teapot");
			scenes.emplace("Teapot", std::move(scn));
		}
		// Fence
		{
			std::unique_ptr<scene> scn(new scene());
			scn->l = std::unique_ptr<light>(new light({0.0f, 8.0f, 7.0f}, glm::normalize(glm::vec3(0.0f, -1.0f, -1.0f)), s));
			scn->objects.emplace_back(new object(floor_vertices, floor_normals, floor_indices, {0.0f, 0.0f, 0.0f}, obj_rot, s));
			scn->objects.emplace_back(new object(fence_vertices, fence_normals, fence_indices, {0.0f, 3.0f, 0.0f}, obj_rot, s));
			scn->cam_pos = {-6.11014f,12.9f,9.67555f};
			scn->cam_theta = 301.0f;
			scn->cam_phi = 121.9f;
			scn->prepare(s);
			scene_order.push_back("Fence");
			scenes.emplace("Fence", std::move(scn));
		}

		decltype(scenes)::iterator current_scene_it;
		glm::vec3 cam_pos;
		float cam_theta;
		float cam_phi;
		auto light_updated(true);

		auto dynamic_light(false);
		auto basic_shadowmap(false);
		auto movement_enabled(false);
		auto kb_fwd(false);
		auto kb_left(false);
		auto kb_right(false);
		auto kb_back(false);

		auto load_scene([&](const std::string &name) {
			current_scene_it = scenes.find(name);
			const auto &scene(current_scene_it->second);
			scene->reset();
			cam_pos = scene->cam_pos;
			cam_theta = scene->cam_theta;
			cam_phi = scene->cam_phi;
			movement_enabled = false;
			kb_fwd = false;
			kb_left = false;
			kb_right = false;
			kb_back = false;
			light_updated = true;
		});

		load_scene("Basic");

		std::vector<std::function<void()>> main_queue; // SDL is effectively not thread safe, so we put calls to it and other things that need to run on the main thread here
		volatile auto do_quit(false);
		auto run_on_main([&](std::function<void()> fn) {
			std::promise<void> p;
			{
				std::lock_guard<std::recursive_mutex> lock(main_mutex);
				main_queue.emplace_back([&fn, &p]() {
					fn();
					p.set_value();
				});
			}
			p.get_future().get();
		});
		cp_thread.emplace([&]() {
			{
				int x, y, w, h;
				run_on_main([&]() {
					SDL_GetWindowPosition(window, &x, &y);
					SDL_GetWindowSize(window, &w, &h);
				});
				std::lock_guard<std::recursive_mutex> lock(main_mutex);
				cp.emplace(nana::rectangle(x + w + 1, y - 30, 300, 400));
				cp->caption("Control Panel");
			}
			nana::place root_layout(*cp);
			root_layout.div("vert<scenes><buffers><settings>");
			nana::group scenes_g(*cp, "Scenes");
			scenes_g.radio_mode(true);
			std::vector<nana::checkbox *> scene_options;
			scene_options.reserve(scenes.size());
			for(auto &scene_name : scene_order) {
				auto &p(*scenes.find(scene_name));
				auto option(&scenes_g.add_option(p.first));
				if(option->caption() == current_scene_it->first) {
					option->check(true);
				}
				scene_options.push_back(option);
			}
			root_layout["scenes"] << scenes_g;
			nana::group buffers_g(*cp, "Buffers");
			buffers_g.radio_mode(true);
			auto &scene_buffer_cb(buffers_g.add_option("Scene"));
			scene_buffer_cb.check(debug_mode == debug_none);
			auto &shadow_map_cb(buffers_g.add_option("Shadow Map"));
			shadow_map_cb.check(debug_mode == debug_shadowmap);
			auto &smoothie_depth_cb(buffers_g.add_option("Smoothie Depth Buffer"));
			smoothie_depth_cb.check(debug_mode == debug_smoothie_depth);
			auto &smoothie_alpha_cb(buffers_g.add_option("Smoothie Alpha Buffer"));
			smoothie_alpha_cb.check(debug_mode == debug_smoothie_alpha);
			root_layout["buffers"] << buffers_g;
			nana::group settings_g(*cp, "Settings");
			settings_g.div("vert<movement_enabled><dynamic_light><basic_shadowmap><<smoothie_size_label><smoothie_size>><<sm_res_label><sm_res>>");
			auto &movement_enabled_cb(*settings_g.create_child<nana::checkbox>("movement_enabled", "Movement Enabled (WASD + Mouse)"));
			movement_enabled_cb.check(movement_enabled);
			auto &dynamic_light_cb(*settings_g.create_child<nana::checkbox>("dynamic_light", "Dynamic Light"));
			dynamic_light_cb.check(dynamic_light);
			auto &basic_shadowmap_cb(*settings_g.create_child<nana::checkbox>("basic_shadowmap", "Basic Shadow Map Only (1 sample)"));
			basic_shadowmap_cb.check(basic_shadowmap);
			settings_g.create_child<nana::label>("smoothie_size_label", "Smoothie Size");
			auto &smoothie_size_sb(*settings_g.create_child<nana::spinbox>("smoothie_size"));
			smoothie_size_sb.range(0.0, 1.0, 0.02);
			smoothie_size_sb.value(std::to_string(smoothie_size));
			settings_g.create_child<nana::label>("sm_res_label", "Shadow Map Res.");
			auto &sm_res_sb(*settings_g.create_child<nana::spinbox>("sm_res"));
			sm_res_sb.range(32, 16384, 128);
			sm_res_sb.value(std::to_string(sm_width));
			root_layout["settings"] << settings_g;
			root_layout.collocate();
			for(auto option : scene_options) {
				option->events().checked([&](const nana::arg_checkbox &msg) {
					if(msg.widget->checked()) {
						std::string caption(msg.widget->caption());
						run_on_main([&]() {
							load_scene(caption);
						});
						{
							std::lock_guard<std::recursive_mutex> lock(main_mutex);
							movement_enabled_cb.check(movement_enabled);
						}
					}
				});
			}
			for(auto option : {&scene_buffer_cb, &shadow_map_cb, &smoothie_depth_cb, &smoothie_alpha_cb}) {
				option->events().checked([&](const nana::arg_checkbox &msg) {
					if(msg.widget->checked()) {
						std::lock_guard<std::recursive_mutex> lock(main_mutex);
						std::string caption(msg.widget->caption());
						if(caption == "Scene") {
							debug_mode = debug_none;
						} else if(caption == "Shadow Map") {
							debug_mode = debug_shadowmap;
						} else if(caption == "Smoothie Depth Buffer") {
							debug_mode = debug_smoothie_depth;
						} else if(caption == "Smoothie Alpha Buffer") {
							debug_mode = debug_smoothie_alpha;
						} else {
							THROW_ERROR();
						}
					}
				});
			}
			movement_enabled_cb.events().checked([&](const nana::arg_checkbox &msg) {
				std::lock_guard<std::recursive_mutex> lock(main_mutex);
				movement_enabled = msg.widget->checked();
				if(!movement_enabled) {
					cam_pos = current_scene_it->second->cam_pos;
					cam_theta = current_scene_it->second->cam_theta;
					cam_phi = current_scene_it->second->cam_phi;
				}
			});
			dynamic_light_cb.events().checked([&](const nana::arg_checkbox &msg) {
				std::lock_guard<std::recursive_mutex> lock(main_mutex);
				dynamic_light = msg.widget->checked();
				if(!dynamic_light) {
					current_scene_it->second->l->pos = current_scene_it->second->orig_light_pos;
					light_updated = true;
				}
			});
			basic_shadowmap_cb.events().checked([&](const nana::arg_checkbox &msg) {
				std::lock_guard<std::recursive_mutex> lock(main_mutex);
				basic_shadowmap = msg.widget->checked();
				light_updated = true;
			});
			smoothie_size_sb.events().text_changed([&](const nana::arg_spinbox &msg) {
				float val;
			 	try {
			 		val = float(msg.widget.to_double());
			 	} catch(const std::invalid_argument &) {
			 		return;
			 	} catch(const std::out_of_range &) {
			 		return;
			 	}
				{
					std::lock_guard<std::recursive_mutex> lock(main_mutex);
					smoothie_size = val;
				}
				run_on_main([&]() {
					light_updated = true;
				});
			 });
			 sm_res_sb.events().text_changed([&](const nana::arg_spinbox &msg) {
				int val;
				try {
					val = msg.widget.to_int();
				} catch(const std::invalid_argument &) {
					return;
				} catch(const std::out_of_range &) {
					return;
				}
				{
					std::lock_guard<std::recursive_mutex> lock(main_mutex);
					sm_width = val;
					sm_height = val;
				}
				run_on_main([&]() {
					XGL(glActiveTexture(GL_TEXTURE0));
					XGL(glBindTexture(GL_TEXTURE_2D, sm_tex));
					XGL(glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, sm_width, sm_height, 0,	GL_RED, GL_FLOAT, nullptr));
					XGL(glBindTexture(GL_TEXTURE_2D, smoothie_depth_tex));
					XGL(glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, sm_width, sm_height, 0, GL_RED, GL_FLOAT, nullptr));
					XGL(glBindTexture(GL_TEXTURE_2D, smoothie_alpha_tex));
					XGL(glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, sm_width, sm_height, 0, GL_RED, GL_FLOAT, nullptr));
					light_updated = true;
				});
			});
			cp->show();
			nana::exec();
		});

		size_t max_frame_times(30);
		std::list<double> frame_times;

		SDL_Event event;
		auto last_tick(clock::now());
		auto tick_rate_ms(16);
		for(;;) {
			auto frame_start_time(clock::now());
			{
				std::lock_guard<std::recursive_mutex> lock(main_mutex);
				if(do_quit) {
					goto quit;
				}
				while(SDL_PollEvent(&event)) {
					switch(event.type) {
						case SDL_KEYDOWN:
						case SDL_KEYUP:
							switch(event.key.keysym.sym) {
								case SDLK_w:
									kb_fwd = event.key.type == SDL_KEYDOWN;
									break;
								case SDLK_a:
									kb_left = event.key.type == SDL_KEYDOWN;
									break;
								case SDLK_d:
									kb_right = event.key.type == SDL_KEYDOWN;
									break;
								case SDLK_s:
								case SDLK_z:
									kb_back = event.key.type == SDL_KEYDOWN;
									break;
							}
							break;
						case SDL_MOUSEMOTION:
							if(movement_enabled && SDL_GetWindowFlags(window) & SDL_WINDOW_INPUT_FOCUS) {
								cam_theta -= event.motion.xrel;
								cam_phi += event.motion.yrel;
								if(cam_theta < 0) {
									cam_theta = 360.0f - cam_theta;
								} else if(cam_theta >= 360.0f) {
									cam_theta = std::fmod(cam_theta, 360.0f);
								}
								if(cam_phi < 0.1f) {
									cam_phi = 0.1f;
								} else if(cam_phi > 179.9f) {
									cam_phi = 179.9f;
								}
							}
							break;
						case SDL_QUIT:
							goto quit;
					}
				}
				for(auto &&fn : main_queue) {
					fn();
				}
				main_queue.clear();

				const auto &scene(current_scene_it->second);

				glm::vec3 cam_dir(std::sin(glm::radians(cam_phi))*std::cos(glm::radians(cam_theta)), std::sin(glm::radians(cam_phi))*std::sin(glm::radians(cam_theta)), std::cos(glm::radians(cam_phi)));

				// run ticks
				auto now(clock::now());
				auto ms_since_tick(std::chrono::duration_cast<std::chrono::milliseconds>(now - last_tick).count());
				if(ms_since_tick >= tick_rate_ms) {
					for(auto ticks(ms_since_tick/tick_rate_ms); ticks > 0; --ticks) {
						if(movement_enabled) {
							std::vector<glm::vec3> move_vecs;
							if(kb_fwd) {
								move_vecs.emplace_back(cam_dir);
							}
							if(kb_left) {
								move_vecs.emplace_back(glm::cross(cam_dir, glm::vec3(0.0f, 0.0f, -1.0f)));
							}
							if(kb_right) {
								move_vecs.emplace_back(glm::cross(cam_dir, glm::vec3(0.0f, 0.0f, 1.0f)));
							}
							if(kb_back) {
								move_vecs.emplace_back(cam_dir * glm::vec3(-1.0f));
							}
							if(!move_vecs.empty()) {
								glm::vec3 mean(0.0f);
								for(auto &&vec : move_vecs) {
									mean += vec;
								}
								mean /= float(move_vecs.size());
								if(glm::length(mean) > 0.0f) {
									mean = glm::normalize(mean);
									auto move_speed(0.5f);
									cam_pos += mean * move_speed;
								}
							}
						}
					}
					last_tick = now;
				}

				auto sm_proj(scene->l->proj(sm_width, sm_height));
				auto cam_proj(glm::perspective(glm::radians(45.0f), float(window_width)/window_height, 1.0f, 100.0f));
				auto sm_view(scene->l->view());
				auto cam_view(glm::lookAt(cam_pos, cam_pos + cam_dir, glm::vec3(0.0f, 0.0f, 1.0f)));

				auto draw_objects([&](const program &object_prog, const glm::mat4 &proj, const glm::mat4 &view, bool sm) {
					auto prog(object_prog.prog);
					XGL(glUseProgram(prog));
					auto proj_loc(glGetUniformLocation(prog, "proj")); XGL_POST();
					auto view_loc(glGetUniformLocation(prog, "view")); XGL_POST();
					auto model_rotate_loc(glGetUniformLocation(prog, "model_rotate")); XGL_POST();
					auto model_translate_loc(glGetUniformLocation(prog, "model_translate")); XGL_POST();
					XGL(glUniformMatrix4fv(proj_loc, 1, GL_FALSE, &proj[0][0]));
					XGL(glUniformMatrix4fv(view_loc, 1, GL_FALSE, &view[0][0]));

					if(!sm) {
						auto light_world_pos_loc(glGetUniformLocation(prog, "light_world_pos")); XGL_POST();
						auto sm_loc(glGetUniformLocation(prog, "sm")); XGL_POST();
						auto sm_proj_loc(glGetUniformLocation(prog, "sm_proj")); XGL_POST();
						auto sm_view_loc(glGetUniformLocation(prog, "sm_view")); XGL_POST();
						XGL(glUniform3fv(light_world_pos_loc, 1, &scene->l->pos[0]));
						XGL(glActiveTexture(GL_TEXTURE0));
						XGL(glBindTexture(GL_TEXTURE_2D, sm_tex));
						XGL(glUniform1i(sm_loc, 0));
						if(!basic_shadowmap) {
							auto smoothie_depth_loc(glGetUniformLocation(prog, "smoothie_depth")); XGL_POST();
							auto smoothie_alpha_loc(glGetUniformLocation(prog, "smoothie_alpha")); XGL_POST();
							XGL(glActiveTexture(GL_TEXTURE1));
							XGL(glBindTexture(GL_TEXTURE_2D, smoothie_depth_tex));
							XGL(glUniform1i(smoothie_depth_loc, 1));
							XGL(glActiveTexture(GL_TEXTURE2));
							XGL(glBindTexture(GL_TEXTURE_2D, smoothie_alpha_tex));
							XGL(glUniform1i(smoothie_alpha_loc, 2));
						}
						XGL(glUniformMatrix4fv(sm_proj_loc, 1, GL_FALSE, &sm_proj[0][0]));
						XGL(glUniformMatrix4fv(sm_view_loc, 1, GL_FALSE, &sm_view[0][0]));
					}

					for(auto &object : scene->objects) {
						auto model_translate(object->model_translate());
						XGL(glUniformMatrix4fv(model_rotate_loc, 1, GL_FALSE, &object->rot[0][0]));
						XGL(glUniformMatrix4fv(model_translate_loc, 1, GL_FALSE, &model_translate[0][0]));
						XGL(glBindVertexArray(object->vao));
						XGL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, object->ibo));
						XGL(glDrawElements(GL_TRIANGLES, object->indices_len, GL_UNSIGNED_INT, reinterpret_cast<GLvoid *>(0)));
					}
				});

				if(dynamic_light) {
					scene->l->pos = scene->orig_light_pos + glm::vec3(std::sin(std::chrono::duration_cast<std::chrono::duration<float, std::ratio<1>>>(clock::now().time_since_epoch()).count()), 0, 0);
					light_updated = true;
				}
				if(light_updated) {
					// render shadow map
					XGL(glBindFramebuffer(GL_FRAMEBUFFER, sm_fb));
					XGL(glViewport(0, 0, sm_width, sm_height));
					XGL(glClearColor(std::numeric_limits<float>::infinity(), 0.0f, 0.0f, 0.0f));
					XGL(glClear(GL_COLOR_BUFFER_BIT));
#if defined(STEP1_SHADOWMAP)
					XGL(glEnable(GL_BLEND));
					XGL(glBlendEquation(GL_MIN));
					draw_objects(*s.sm_object_prog, sm_proj, sm_view, true);
					XGL(glDisable(GL_BLEND));
#endif

					if(!basic_shadowmap) {
						// construct smoothie geometry
						for(auto &buf : scene->l_smoothie_buffers) {
							buf->build();
						}
						// render smoothies
						XGL(glBindFramebuffer(GL_FRAMEBUFFER, smoothie_fb));
						XGL(glViewport(0, 0, sm_width, sm_height));
						XGL(glDrawBuffer(GL_COLOR_ATTACHMENT0));
						XGL(glClearColor(std::numeric_limits<float>::infinity(), 0.0f, 0.0f, 0.0f));
						XGL(glClear(GL_COLOR_BUFFER_BIT));
						XGL(glDrawBuffer(GL_COLOR_ATTACHMENT1));
						XGL(glClearColor(1.0f, 0.0f, 0.0f, 0.0f));
						XGL(glClear(GL_COLOR_BUFFER_BIT));
						{
							GLenum bufs[] = {
								GL_COLOR_ATTACHMENT0,
								GL_COLOR_ATTACHMENT1
							};
							XGL(glDrawBuffers((sizeof bufs)/(sizeof bufs[0]), bufs));
						}
#if defined(STEP1_SHADOWMAP) && defined(STEP2_SMOOTHIES)
						XGL(glEnable(GL_BLEND));
						XGL(glBlendEquation(GL_MIN));
						{
							auto prog_edges(s.smoothie_edges_prog->prog);
							auto edges_sm_loc(glGetUniformLocation(prog_edges, "sm")); XGL_POST();
							auto prog_corners(s.smoothie_corners_prog->prog);
							auto corners_smoothie_size_loc(glGetUniformLocation(prog_corners, "smoothie_size")); XGL_POST();
							auto corners_sm_loc(glGetUniformLocation(prog_corners, "sm")); XGL_POST();
							XGL(glActiveTexture(GL_TEXTURE0));
							XGL(glBindTexture(GL_TEXTURE_2D, sm_tex));
							// render edges
							XGL(glUseProgram(prog_edges));
							XGL(glUniform1i(edges_sm_loc, 0));
							for(auto &buf : scene->l_smoothie_buffers) {
								XGL(glBindVertexArray(buf->edges_vao));
								XGL(glDrawArrays(GL_TRIANGLES, 0, buf->num_edge_vertices));
							}
							// render corners
							XGL(glUseProgram(prog_corners));
							XGL(glUniform1f(corners_smoothie_size_loc, smoothie_size));
							XGL(glUniform1i(corners_sm_loc, 0));
							for(auto &buf : scene->l_smoothie_buffers) {
								XGL(glBindVertexArray(buf->corners_vao));
								XGL(glDrawArrays(GL_TRIANGLES, 0, buf->num_corner_vertices));
							}
						}
						XGL(glDisable(GL_BLEND));
#endif
					}
					light_updated = false;
				}

				// render scene
				XGL(glBindFramebuffer(GL_FRAMEBUFFER, 0));
				XGL(glEnable(GL_DEPTH_TEST));
				XGL(glViewport(0, 0, window_width, window_height));
				XGL(glClearColor(0.0f, 0.0f, 0.0f, 1.0f));
				XGL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

				if(debug_mode != debug_none) {
					auto prog(s.tex_debug_prog->prog);
					XGL(glUseProgram(prog));
					auto mode_loc(glGetUniformLocation(prog, "mode")); XGL_POST();
					auto tex_loc(glGetUniformLocation(prog, "tex")); XGL_POST();
					XGL(glActiveTexture(GL_TEXTURE0));
					switch(debug_mode) {
					case debug_shadowmap:
						XGL(glUniform1i(mode_loc, 0));
						XGL(glBindTexture(GL_TEXTURE_2D, sm_tex));
						break;
					case debug_smoothie_depth:
						XGL(glUniform1i(mode_loc, 0));
						XGL(glBindTexture(GL_TEXTURE_2D, smoothie_depth_tex));
						break;
					case debug_smoothie_alpha:
						XGL(glUniform1i(mode_loc, 1));
						XGL(glBindTexture(GL_TEXTURE_2D, smoothie_alpha_tex));
						break;
					default:
						THROW_ERROR();
					}
					XGL(glUniform1i(tex_loc, 0));
					XGL(glBindVertexArray(screen_vao));
					XGL(glDrawArrays(GL_TRIANGLES, 0, (sizeof tex_debug_vertices)/(3*(sizeof tex_debug_vertices[0]))));
				} else {
					draw_objects(basic_shadowmap ? *s.cam_object_basic_prog : *s.cam_object_prog, cam_proj, cam_view, false);

					// render scene light position
					{
						auto prog(s.cam_light_prog->prog);
						XGL(glUseProgram(prog));
						auto proj_loc(glGetUniformLocation(prog, "proj")); XGL_POST();
						auto view_loc(glGetUniformLocation(prog, "view")); XGL_POST();
						auto model_loc(glGetUniformLocation(prog, "model")); XGL_POST();
						XGL(glUniformMatrix4fv(proj_loc, 1, GL_FALSE, &cam_proj[0][0]));
						XGL(glUniformMatrix4fv(view_loc, 1, GL_FALSE, &cam_view[0][0]));
						auto light_model(glm::translate(scene->l->pos));
						XGL(glUniformMatrix4fv(model_loc, 1, GL_FALSE, &light_model[0][0]));
						XGL(glBindVertexArray(scene->l->vao));
						XGL(glDrawArrays(GL_POINTS, 0, (sizeof light_vertices)/(3*(sizeof light_vertices[0]))));
					}
				}

				XGL(glDisable(GL_DEPTH_TEST));

				{
					std::ostringstream oss;
					auto ms(std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(clock::now() - frame_start_time).count());
					if(frame_times.size() > max_frame_times) {
						frame_times.pop_front();
					}
					frame_times.push_back(ms);
					auto avg(std::accumulate(frame_times.begin(), frame_times.end(), 0.0)/frame_times.size());
					size_t num_edges(0);
					for(const auto &buf : scene->l_smoothie_buffers) {
						num_edges += buf->edges.size();
					}
					oss << "Render - " << num_edges << " edges - " << avg << "ms";
					SDL_SetWindowTitle(window, oss.str().c_str());
				}
			}
			SDL_GL_SwapWindow(window);
			std::this_thread::yield();
		}
quit :
		;
	} catch(const std::exception &e) {
		std::cerr << "Error: " << e.what() << std::endl;
#if defined(DEBUG)
		__debugbreak();
#endif
		error = true;
	}
	if(cp_thread) {
		std::lock_guard<std::recursive_mutex> main_lock(main_mutex);
		if(cp) {
			nana::internal_scope_guard nana_lock;
			cp->close();
		}
		cp_thread->join();
	}
	return error ? 1 : 0;
}
