#include<iostream>
#include<GL/glew.h>
#include<GL/freeglut.h>
#include<glm/vec3.hpp>
#include<glm/mat4x4.hpp>
#include<glm/gtx/transform.hpp>

#include<Magick++.h>

#define ToRadian(x) ((x) * 3.14159265359 / 180.0f)

GLuint VBO;
GLuint IBO;
GLuint gwl;
GLuint gSampler;
GLuint GLtexture;

static const char* pVS = "                                                          \n\
#version 330                                                                        \n\
                                                                                    \n\
layout (location = 0) in vec3 Position;                                             \n\
layout (location = 1) in vec2 TexCoord;                                             \n\
                                                                                    \n\
uniform mat4 gWVP;                                                                  \n\
                                                                                    \n\
out vec2 TexCoord0;                                                                 \n\
                                                                                    \n\
void main()                                                                         \n\
{                                                                                   \n\
    gl_Position = gWVP * vec4(Position, 1.0);                                       \n\
    TexCoord0 = TexCoord;                                                           \n\
}";

static const char* pFS = "                                                          \n\
#version 330                                                                        \n\
                                                                                    \n\
in vec2 TexCoord0;                                                                  \n\
                                                                                    \n\
out vec4 FragColor;                                                                 \n\
                                                                                    \n\
uniform sampler2D gSampler;                                                         \n\
                                                                                    \n\
void main()                                                                         \n\
{                                                                                   \n\
    FragColor = texture2D(gSampler, TexCoord0.xy);                                  \n\
}";

unsigned int Indices[] = { 0, 3, 1,
                           1, 3, 2,
                           2, 3, 0,
                           0, 2, 1 };

float scale = 0.000f;
const float ar = 1024 / 768;
const float zNear = 1.0f;
const float zFar = 1000.0f;
const float zRange = zNear - zFar;

struct Vertex {
    glm::vec3 fst;
    glm::vec2 snd;

    Vertex(glm::vec3 inp1, glm::vec2 inp2) {
        fst = inp1;
        snd = inp2;
    }
};

static void addShader(GLuint share_prog, const char* text_shared, GLenum type)
{
    GLuint obj = glCreateShader(type);
    if (obj == 0) {
        fprintf(stderr, "Error creating shader type %d\n", type);
        exit(0);
    }

    const GLchar* p[1];
    p[0] = text_shared;

    GLint len[1];
    len[0] = strlen(text_shared);

    glShaderSource(obj, 1, p, len);
    glCompileShader(obj);

    GLint conn;
    glGetShaderiv(obj, GL_COMPILE_STATUS, &conn);

    if (!conn) {
        printf("We can not connect");
        exit(1);
    }

    glAttachShader(share_prog, obj);
}


void RenderSceneCB()
{
    glClear(GL_COLOR_BUFFER_BIT);
    scale += 0.0125f;

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), 0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (const GLvoid*)12);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IBO);

    //Triangle move
    glm::mat4 myMatrixMove(1.0f, 0.0f, 0.0f, 0.0f,
                           0.0f, 1.0f, 0.0f, 0.0f,
                           0.0f, 0.0f, 1.0f, 0.0f,
                           0.0f, 0.0f, sinf(scale), 1.0f);
    //Triangle rotation
    glm::mat4 myMatrixRotateY(cosf(scale), 0.0f, -sinf(scale), 0.0f,
                              0.0f, 1.0f, 0.0f, 0.0f,
                              sinf(scale), 0.0f, cosf(scale), 0.0f,
                              0.0f, 0.0f, 0.0f, 1.0f);
    glm::mat4 myMatrixPersProj = glm::perspective(30.0f, ar, zNear, zFar);
    glm::mat4 ViewMatrix = glm::lookAt(glm::vec3(0.0, 0.0, -2.0),
                                       glm::vec3(0.0, 0.0, 0.0),
                                       glm::vec3(1.0, -2.0, 1.0));
    glm::mat4 myMatrixTransformation = myMatrixPersProj * ViewMatrix * myMatrixRotateY;
    glUniformMatrix4fv(gwl, 1, GL_TRUE, (const GLfloat*)&myMatrixTransformation);

    glDrawElements(GL_TRIANGLES, 12, GL_UNSIGNED_INT, 0);
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);

    glutSwapBuffers();
}

int main(int argc, char** argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);

    glutInitWindowSize(1280, 768);
    glutInitWindowPosition(100, 100);

    glutCreateWindow("Lab Work 3");
    glutDisplayFunc(RenderSceneCB);
    glutIdleFunc(RenderSceneCB);

    GLenum res = glewInit();
    if (res != GLEW_OK) {
        fprintf(stderr, "Error: GLEW INCORRECT'\n");
        return 1;
    }

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glFrontFace(GL_CW);
    glCullFace(GL_BACK);
    glEnable(GL_CULL_FACE);

    Vertex input[4] = {
        Vertex(glm::vec3 {-1.0f, -1.0f, 0.5f}, glm::vec2 {0.0f, 0.0f}),
        Vertex(glm::vec3 {0.0f, -1.0f, -1.0f}, glm::vec2 {0.0f, 0.0f}),
        Vertex(glm::vec3 {1.0f, -1.0f, 0.5f}, glm::vec2 {1.0f, 0.0f}),
        Vertex(glm::vec3 {0.0f, 1.0f, 0.0f}, glm::vec2 {0.0f, 1.0f}),
    };

    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(input), input, GL_STATIC_DRAW);

    glGenBuffers(1, &IBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(Indices), Indices, GL_STATIC_DRAW);

    GLuint shader_program = glCreateProgram();

    if (shader_program == 0) {
        printf("shader prog is not created.\n");
        return -1;
    }

    addShader(shader_program, pVS, GL_VERTEX_SHADER);
    addShader(shader_program, pFS, GL_FRAGMENT_SHADER);
    glLinkProgram(shader_program);
    glUseProgram(shader_program);
    gwl = glGetUniformLocation(shader_program, "gWVP");
    gSampler = glGetUniformLocation(shader_program, "gSampler");

    if (gwl == 0xFFFFFFFF) {
        printf("Error with not ");
        return -1;
    }

    glUniform1i(gSampler, 0);
    Magick::InitializeMagick(nullptr);

    Magick::Image image("Test.jpg");
    Magick::Blob blob;

    image.magick("RGBA");
    image.write(&blob);

    glGenTextures(1, &GLtexture);
    glBindTexture(GL_TEXTURE_2D, GLtexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    gluBuild2DMipmaps(GL_TEXTURE_2D, 4, image.columns(), image.rows(), GL_RGBA, GL_UNSIGNED_BYTE, blob.data());

    glutMainLoop();
}