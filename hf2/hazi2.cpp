
#include "framework.h"

const vec3 points[20] =
{
	vec3(0, 0.618, 1.618),vec3(0, -0.618, 1.618),vec3(0, -0.618, -1.618),vec3(0, 0.618, -1.618),
	vec3(1.618 , 0, 0.618),vec3(-1.618 , 0, 0.618),vec3(-1.618 , 0, -0.618),vec3(1.618, 0, -0.618),
	vec3(0.618, 1.618, 0),vec3(-0.618, 1.618, 0),vec3(-0.618, -1.618, 0),vec3(0.618, -1.618, 0),
	vec3(1, 1, 1),vec3(-1, 1, 1),vec3(-1, -1, 1),vec3(1, -1, 1),vec3(1, -1, -1),vec3(1, 1, -1),vec3(-1, 1, -1),vec3(-1, -1, -1),
};

const int faces[12][5] =
{
	{1, 2, 16, 5, 13},
	{1, 13, 9, 10, 14},
	{1, 14, 6, 15, 2},
	{2, 15, 11, 12, 16},
	{3, 4, 18, 8, 17},
	{3, 17, 12, 11, 20},
	{3, 20, 7, 19, 4},
	{19, 10, 9, 18, 4},
	{16, 12, 17, 8, 5},
	{5, 8, 18, 9, 13},
	{14, 10, 19, 7, 6},
	{6, 7, 20, 11, 15}
};

const float epsilon = 0.01f;


struct Material {
	int id;
	vec3 ka, kd, ks;
	float  shininess;
	Material(int _id, vec3 _kd, vec3 _ks, float _shininess) : id(_id), ka(_kd* M_PI), kd(_kd), ks(_ks) { shininess = _shininess; }
};


struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir, weight;
	Ray(vec3 _start, vec3 _dir, vec3 _weight) { start = _start; dir = normalize(_dir); weight = _weight; }
};

class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray, int portaldepth) = 0;
};

std::vector<Intersectable*> objects;

struct Dodecahedron : public Intersectable {
	const int numfaces = 12;
	float dodesize;
	Dodecahedron(float size, Material* _material) {
		dodesize = size;
		material = _material;
	}

	Hit firstIntersect(Ray ray, int depth = 0) {
		Hit bestHit;
		if (depth > 4) return bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray, depth);
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	void getDodePlain(int i, float size, vec3& p, vec3& normal) {
		vec3 p1 = points[faces[i][0] - 1], p2 = points[faces[i][1] - 1], p3 = points[faces[i][2] - 1];
		normal = cross(p2 - p1, p3 - p1);
		if (dot(p1, normal) < 0) normal = -normal;
		p = p1 * size + vec3(0, 0, epsilon);
	}

	Hit intersect(const Ray& ray, int portaldepth) {
		Hit hit;
		if (portaldepth >= 5) return hit;
		for (int i = 0; i < numfaces; i++)
		{
			vec3 p1, normal;
			getDodePlain(i, dodesize, p1, normal);
			float t = abs(dot(normal, ray.dir)) > epsilon ? dot(p1 - ray.start, normal) / dot(normal, ray.dir) : -1;
			if (t <= epsilon || (t > hit.t && hit.t > 0)) continue;
			vec3 pinter = ray.start + ray.dir * t;
			bool outs = false;
			bool outs2 = false;
			int whichface = 0;
			for (int j = 0; j < numfaces; j++)
			{
				if (i == j) continue;
				vec3 p12, normal2;
				getDodePlain(j, dodesize - 0.1, p12, normal2);
				if (dot(normal2, pinter - p12) > 0) outs = true;
				getDodePlain(j, dodesize, p12, normal2);
				if (dot(normal2, pinter - p12) > 0) {
					outs2 = true;
					whichface = j;
					break;
				}
			}
			if (outs && !outs2) {
				hit.t = t;
				hit.normal = normalize(normal);
				hit.position = pinter;
				hit.material = material;
				break;
			}
			else if (!outs && !outs2) {
				vec3 kozepp;
				for (int i = 0; i < 5; i++)
				{
					kozepp = kozepp + points[faces[whichface][i] - 1];
				}
				kozepp = kozepp / 5;
				pinter = pinter - kozepp;
				float fok = 72 * M_PI / 180;
				vec3 v = pinter * cosf(fok) + cross(normal, pinter) * sinf(fok) + normal * dot(pinter, normal) * (1 - cosf(fok)) + kozepp;
				vec3 rdir = ray.dir - normal * dot(normal, ray.dir) * 2.0f;
				rdir = rdir * cosf(fok) + cross(normal, rdir) * sinf(fok) + normal * dot(normal, rdir) * (1 - cosf(fok));
				return firstIntersect(Ray(v, rdir, ray.weight), portaldepth + 1);
			}
		}
		return hit;
	}
};

struct GoldObj : public Intersectable {
	GoldObj(Material* _material) {
		material = _material;
	}
	float distance(vec3 a) {
		return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
	}

	vec3 gradf(vec3 v) {
		return vec3(0.1 * 2 * v.x, 0.2 * 2* v.y, -0.3);
	}

	Hit intersect(const Ray& ray, int portaldepth) {
		Hit hit;
		float d = sqrtf(powf(0.2 * ray.start.x * ray.dir.x + 0.4 * ray.start.y * ray.dir.y - 0.3 * ray.dir.z ,2) - 4 * (0.1 * ray.dir.x * ray.dir.x + 0.2 * ray.dir.y * ray.dir.y) * (0.1 * ray.start.x * ray.start.x + 0.2 * ray.start.y * ray.start.y - 0.3 *	 ray.start.z));
		if (d < 0) return hit;
		float t = -1;
		if (d == 0) {
			float t1 = (-0.2 * ray.dir.x * ray.start.x - 0.4 * ray.dir.y * ray.start.y + 0.3 * ray.dir.z) / (2 * (0.1 * ray.dir.x * ray.dir.x + 0.2 * ray.dir.y * ray.dir.y));
			if (t1 > 0 && distance(ray.start + ray.dir * t1) <= 0.3)
				t = t1;
		}
		else {
			float t1 = ((-0.2 * ray.dir.x * ray.start.x - 0.4 * ray.dir.y * ray.start.y + 0.3 * ray.dir.z) + d) / (2 * (0.1 * ray.dir.x * ray.dir.x + 0.2 * ray.dir.y * ray.dir.y));
			float t2 = ((-0.2 * ray.dir.x * ray.start.x - 0.4 * ray.dir.y * ray.start.y + 0.3 * ray.dir.z) - d) / (2 * (0.1 * ray.dir.x * ray.dir.x + 0.2 * ray.dir.y * ray.dir.y));
			if (t1 < t2) {
				if (t1 > 0 && distance(ray.start + ray.dir * t1) <= 0.3)t = t1;
				else if(t2 > 0 && distance(ray.start + ray.dir * t2) <= 0.3)t = t2;
			}
			else {
				if (t2 > 0 && distance(ray.start + ray.dir * t2) <= 0.3)t = t2;
				else if (t1 > 0 && distance(ray.start + ray.dir * t1) <= 0.3)t = t1;
			}
		}
		hit.t = t;
		hit.position = ray.start + ray.dir * t;
		hit.normal = normalize(gradf(hit.position));
		hit.material = material;

		return hit;
	}

};

class Camera {
	vec3 eye, lookat, right, up;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir, vec3(1, 1, 1));
	}
	void Animate(float t) {
		float r = sqrtf(eye.x * eye.x + eye.y * eye.y);
		eye = vec3(r * cos(t) + lookat.x, r * sin(t) + lookat.y, eye.z);
		set(eye, lookat, vec3(0, 1, 0), 45 * M_PI / 180);
	}

};

float F(float n, float k) { return ((n - 1) * (n - 1) + k * k) / ((n + 1) * (n + 1) + k * k); }
vec3 F0 = vec3(F(0.17, 3.1), F(0.35, 2.7), F(1.5, 1.9));
struct Light {
	vec3 position;
	vec3 Le;

	void set(vec3 _position, vec3 _Le) {
		position = normalize(_position);
		Le = _Le;
	}
};

class Scene {
	Light light;
	Camera camera;
	vec3 La;
public:
	void build() {
		vec3 eye = vec3(1.0, 0.6, 0.7), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.678f, 0.847f, 1.0f);
		vec3 lightPos(0, 0.5, 1.4), Le(2, 2, 2);
		light.set(lightPos, Le);

		vec3 kd(0.4f, 0.2f, 0.1f), ks(2, 2, 2);
		Material* material = new Material(1, kd, ks, 50);
		Material* gold = new Material(2, vec3(0.17, 0.35, 1.5), vec3(3.1, 2.7, 1.9), 50);
		objects.push_back(new Dodecahedron((sqrtf(3) * 4) / ((1 + sqrtf(5)) * sqrtf(3)), material));//https://en.wikipedia.org/wiki/Regular_dodecahedron
		objects.push_back(new GoldObj(gold));
	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray, int depth = 0) {
		Hit bestHit;
		if (depth > 5) return bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray, depth);
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}
	
	vec3 trace(Ray ray, int depth = 0) {
		if (depth > 5) return La;
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return La;

		if (hit.material->id == 2) {
			float cosa = -dot(ray.dir, hit.normal);
			vec3 F = F0 + (vec3(1, 1, 1) - F0) * powf(1.0 - cosa, 5);
			vec3 refldir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0;
			return trace(Ray(hit.position + hit.normal * epsilon, refldir, depth + 1)) * F;
		}
		else {
			vec3 outRadiance = hit.material->ka * La;

			vec3 lightdir = normalize(light.position - hit.position + hit.normal * epsilon);

			float cosTheta = dot(hit.normal, lightdir);
			if (cosTheta > 0) {
				outRadiance = outRadiance + light.Le * hit.material->kd * cosTheta;
				vec3 halfway = normalize(-ray.dir + lightdir);
				float cosDelta = dot(hit.normal, halfway);
				if (cosDelta > 0) outRadiance = outRadiance + light.Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
			}
			return outRadiance;
		}
		
	}

	void Animate(float t) {
		camera.Animate(t);
	}
};

const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;						
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		
	}
)";

const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;		
	out vec4 fragmentColor;

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

GPUProgram gpuProgram;
Scene scene;
class FullScreenTexturedQuad {
	unsigned int vao;
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image)
	{
		unsigned int vbo;
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	}

	void Draw() {
		glBindVertexArray(vao);
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;
// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();
	std::vector<vec4> image(windowWidth * windowHeight);
	scene.render(image);
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}
// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	delete fullScreenTexturedQuad;
	std::vector<vec4> image(windowWidth * windowHeight);
	scene.render(image);
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glutSwapBuffers();
}
// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {}
// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {}
// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {}
// Mouse click event
void onMouse(int button, int state, int pX, int pY) {}
// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	scene.Animate(glutGet(GLUT_ELAPSED_TIME) / 1000.0f);
	glutPostRedisplay();
}
