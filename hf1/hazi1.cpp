
#include "framework.h"

class ImmediateModeRenderer2D : public GPUProgram {
	const char* const vertexSource = R"(
		#version 330
		precision highp float;
		layout(location = 0) in vec2 vertexPosition;	// Attrib Array 0

		void main() { gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1); }	
	)";

	const char* const fragmentSource = R"(
		#version 330
		precision highp float;
		uniform vec3 color;
		out vec4 fragmentColor;	

		void main() { fragmentColor = vec4(color, 1); }
	)";

	unsigned int vao, vbo;

public:
	ImmediateModeRenderer2D() {
		glViewport(0, 0, windowWidth, windowHeight);
		glLineWidth(2.0f); glPointSize(10.0f);

		create(vertexSource, fragmentSource, "outColor");
		glGenVertexArrays(1, &vao); glBindVertexArray(vao);
		glGenBuffers(1, &vbo); 		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), NULL);
	}

	void DrawGPU(int type, std::vector<vec2> vertices, vec3 color) {
		setUniform(color, "color");
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vec2), &vertices[0], GL_DYNAMIC_DRAW);
		glDrawArrays(type, 0, vertices.size());
	}

	~ImmediateModeRenderer2D() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};

ImmediateModeRenderer2D* renderer;

const int vertex = 50;
const float telitetseg = 0.05;
const int nTesselatedVertices = 20;
const int edges = (vertex - 1) * vertex / 2 * telitetseg;
const int dcsillag = 1;
const float deltat = 0.01;

class Graph {
	int neighmatrix[vertex][vertex] = {};
	vec2 vertexs[vertex];
	vec2 fvertexs[vertex];
	vec2 vVertexs[vertex];

	float distance(vec2 a, vec2 b) {
		float w1 = sqrt(pow(a.x, 2) + pow(a.y, 2) + 1);
		float w2 = sqrt(pow(b.x, 2) + pow(b.y, 2) + 1);
		float lorentz = a.x * b.x + a.y * b.y - w1 * w2;
		return acosh(-1.0 * lorentz);
	}

	vec2 calculateForce(int whichvertex) {
		vec2 force(0, 0);
		for (int i = 0; i < vertex; i++)
		{
			if (i != whichvertex) {
				float distance = this->distance(vertexs[whichvertex], vertexs[i]);
				if (neighmatrix[whichvertex][i] == 1) {
					if (dcsillag < distance) {
						force = force + (vertexs[i] - vertexs[whichvertex]) * pow(distance, 2);
					}
					else if (dcsillag > distance) {
						force = force + (vertexs[whichvertex] - vertexs[i]) * pow(1 / distance, 2);
					}
					else {
						return force;
					}
				}
				else {
					force = force + (vertexs[whichvertex] - vertexs[i]) * 1.5 * (pow(1 / distance, 2) - 0.2);
				}
			}
		}
		return force;
	}

public:
	Graph() {
		
		int currentedges = 0;
		int i = 1;
		srand(1111);

		while (currentedges != edges){
			int ran1 = rand() % vertex;
			int ran2 = rand() % vertex;
			i++;
			if (ran1 == ran2) { continue; }
			if (neighmatrix[ran1][ran2] == 1) { continue; }

			neighmatrix[ran1][ran2] = 1;
			neighmatrix[ran2][ran1] = 1;
			currentedges += 1;
		}

		for (int j = 0; j < 5; j++)
		{
			for (int k = 0; k < 10; k++)
			{
				vertexs[j * 10 + k] = vec2((float)j / 10 - 0.25f, (float)k / 10 - 0.5f);
			}
		}
		for (int i = 0; i < vertex; i++)
		{
			fvertexs[i] = calculateForce(i);
		}

	}
	
	void fineLayout(){
		float minpot = -1;
		for (int i = 0; i < 200; i++)
		{
			vec2 tempvertexs[vertex];
			for (int j = 0; j < vertex; j++)
			{
					tempvertexs[j] = vec2(float(rand() / float(RAND_MAX)) * 2 - 1, float(rand() / float(RAND_MAX)) * 2 - 1);
			}
			
			for (int i = 0; i < 5; i++)
			{
				for (int j = 0; j < vertex; j++)
				{
					vec2 temptk;
					for (int k = 0; k < vertex; k++)
					{
						if (k == j) continue;
						if (neighmatrix[k][j] == 1) {
							temptk = temptk + tempvertexs[k];
						}
						else {
							temptk = temptk - tempvertexs[k];
						}
					}
					tempvertexs[j] = temptk/5;
				}
			}
			for (int i = 0; i < vertex; i++)
			{
					fvertexs[i] = calculateForce(i);
			}

			vec2 pote;
			for (int f = 0; f < vertex; f++)
			{
				pote = pote + calculateForce(f);
			}
			if (minpot == -1 || sqrt(pow(pote.x, 2) + pow(pote.y, 2)) < minpot) {
				minpot = sqrt(pow(pote.x, 2) + pow(pote.y, 2));
				for (int v = 0; v < vertex; v++)
				{
					vertexs[v] = tempvertexs[v];
				}
			}
		}

		for (int i = 0; i < vertex; i++)
		{
			fvertexs[i] = calculateForce(i);
		}

	}

	void simRound() {
		vec2 tempvertexs[vertex];
		for (int i = 0; i < vertex; i++)
		{
			fvertexs[i] = fvertexs[i] - vVertexs[i];
			vVertexs[i] = fvertexs[i] * deltat;
			tempvertexs[i] = vertexs[i] + vVertexs[i] * deltat;
		}
		for (int i = 0; i < vertex; i++)
		{
			vertexs[i] = tempvertexs[i];
		}
	}

	void eltolas(vec2 a) {
		if (a.x == 0 && a.y == 0)return;
		a = a / sqrt(1 - a.x * a.x - a.y * a.y);
		

		float d = distance(a, vec2(0, 0));
		vec2 v = (a - vec2(0, 0) * cosh(d)) / sinh(d);
		vec2 m1 = vec2(0, 0) * cosh(d / 4) + v * sinh(d / 4);
		vec2 m2 = vec2(0, 0) * cosh(3*d / 4) + v * sinh(3 * d / 4);
		for (int i = 0; i < vertex; i++)
		{
			vec2 v1 = (m1 - vertexs[i] * cosh(distance(vertexs[i], m1))) / sinh(distance(vertexs[i], m1));
			vertexs[i] = vertexs[i] * cosh(2 * distance(vertexs[i], m1)) + v1 * sinh(2 * distance(vertexs[i], m1));
			vec2 v2 = (m2 - vertexs[i] * cosh(distance(vertexs[i], m2))) / sinh(distance(vertexs[i], m2));
			vertexs[i] = vertexs[i] * cosh(2 * distance(vertexs[i], m2)) + v2 * sinh(2 * distance(vertexs[i], m2));
		}
	}

	void Draw() {
		for (int i = 0; i < vertex; i++)
		{
			for (int j = 0; j < vertex; j++)
			{
				if (i < j && neighmatrix[i][j] == 1) {
					std::vector<vec2> betweenVertexline;
					betweenVertexline.push_back(vertexs[i]/ sqrt(pow(vertexs[i].x, 2) + pow(vertexs[i].y, 2) + 1));
					betweenVertexline.push_back(vertexs[j]/ sqrt(pow(vertexs[j].x, 2) + pow(vertexs[j].y, 2) + 1));
					renderer->DrawGPU(GL_LINE_STRIP, betweenVertexline, vec3(1.0f, 1.0f, 0.0f));
				}
			}
		}
		for (int i = 0; i < vertex; i++)
		{
			std::vector<vec2> vertexCirclePoints;
			std::vector<vec2> vertexCircleColor1;
			std::vector<vec2> vertexCircleColor2;
			for (int j = 0; j < nTesselatedVertices; j++) {
				float phi = j * 2.0f * M_PI / nTesselatedVertices;
				vec2 kp = (vertexs[i] / sqrt(vertexs[i].x * vertexs[i].x + vertexs[i].y * vertexs[i].y + 1) + vec2(cosf(phi), sinf(phi)) * 0.1f);
				kp = calculateHypePoint(vertexs[i], kp, 0.05f);
				vertexCirclePoints.push_back(kp / sqrt(kp.x * kp.x + kp.y * kp.y + 1));
			}

			vec2 rect[4] = { vec2(0.0f, -0.1f), vec2(0.0f, 0.1f), vec2(0.1f, 0.1f), vec2(0.1f, -0.1f) };

			for (int j = 0; j < 4; j++)
			{
				vec2 kp = (vertexs[i] / sqrt(vertexs[i].x * vertexs[i].x + vertexs[i].y * vertexs[i].y + 1) + rect[j]);
				vec2 p = calculateHypePoint(vertexs[i], kp, 0.03f);
				vertexCircleColor1.push_back(p / sqrt(p.x * p.x + p.y * p.y + 1));
				p = calculateHypePoint(vertexs[i], kp, -0.03f);
				vertexCircleColor2.push_back(p / sqrt(p.x * p.x + p.y * p.y + 1));
			}

			renderer->DrawGPU(GL_TRIANGLE_FAN, vertexCirclePoints, vec3(0.5f, 0.5f, 0.5f));
			renderer->DrawGPU(GL_TRIANGLE_FAN, vertexCircleColor1, vec3((i & (1 << 5)) >> 5, (i & (1 << 4)) >> 4, (i & (1 << 3)) >> 3));
			renderer->DrawGPU(GL_TRIANGLE_FAN, vertexCircleColor2, vec3((i & (1 << 2)) >> 2, (i & (1 << 1)) >> 1, (i & (1 <<0)) >> 0));
		}
	}

	vec2 calculateHypePoint(vec2 p, vec2 q,float size) {
		q = q / sqrt(1 - q.x * q.x - q.y * q.y);
		vec2 v = (q - p * cosh(distance(q, p))) / sinh(distance(q, p));
		return p * cosh(size) + v * sinh(size);
	}
};

Graph graph;
vec2 clicked;
int clickstate;
// Initialization, create an OpenGL context
void onInitialization() {
	renderer = new ImmediateModeRenderer2D();
	graph = Graph();
}
// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
	graph.Draw();
	glutSwapBuffers();
}
// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == ' ') { 
		graph.fineLayout();
		glutPostRedisplay(); 
	}
}
// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {}
// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	if (clickstate == GLUT_DOWN) {
		float cX = 2.0f * pX / windowWidth - 1;
		float cY = 1.0f - 2.0f * pY / windowHeight;
		graph.eltolas(vec2(cX, cY) - clicked);
		clicked = vec2(cX, cY);
		glutPostRedisplay();
	}	
}
// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	if (1 - cX * cX - cY * cY < 0)return;

	if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) {
		clicked = vec2(cX, cY);
		clickstate = GLUT_DOWN;
	}else {
		clickstate = GLUT_UP;
	}
}
// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	graph.simRound();
	glutPostRedisplay();
}