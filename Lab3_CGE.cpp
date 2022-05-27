#include <GL/glew.h>
#include <GL/freeglut.h>
#include "glm/glm.hpp"
#include "Magick++.h"
#include <iostream>

#define M_PI 3.14159265358979323846 
#define ToRadian(x) ((x) * M_PI / 180.0f) 
#define ToDegree(x) ((x) * 180.0f / M_PI) 

namespace VectorsMath {
    const int x = 0, y = 1, z = 2;

    glm::vec3 Cross(glm::vec3& v1, glm::vec3& v2) {
        const float _x = v1[x] * v2[z] - v1[z] * v2[y];
        const float _y = v1[z] * v2[x] - v1[x] * v2[z];
        const float _z = v1[x] * v2[y] - v1[y] * v2[x];

        return glm::vec3{ _x, _y, _z };
    }

    void Normalize(glm::vec3& v) {
        const float Length = sqrtf(v[x] * v[x] + v[y] * v[y] + v[z] * v[z]);

        v[x] /= Length;
        v[y] /= Length;
        v[z] /= Length;
    }
};

float scale = 0.01f;
const float WINDOW_HEIGHT = 1024;
const float WINDOW_WIDTH = 768;
const int MAX_POINT_LIGHTS = 3;
const int MAX_SPOT_LIGHTS = 2;
GLint success;
GLchar InfoLog[1024];

using namespace glm;
using namespace std;
using namespace Magick;

struct Vertex
{
    vec3 m_pos;
    vec2 m_tex;
    vec3 m_norm;

    Vertex(vec3 pos, vec2 tex)
    {
        m_pos = pos;
        m_tex = tex;
        m_norm = vec3(0.0f, 0.0f, 0.0f);
    }
};
struct BaseLight
{
    vec3 Color;
    float AmbientIntensity;
    float DiffuseIntensity;
    BaseLight()
    {
        Color = vec3(0.0f, 0.0f, 0.0f);
        AmbientIntensity = 0.0f;
        DiffuseIntensity = 0.0f;
    }
};
struct DirectionalLight : public BaseLight
{
    vec3 Direction;
    DirectionalLight()
    {
        Direction = vec3(0.0f, 0.0f, 0.0f);
    }
};
struct PointLight : public BaseLight
{
    vec3 Position;
    struct
    {
        float Constant;
        float Linear;
        float Exp;
    } Attenuation;

    PointLight()
    {
        Position = vec3(0.0f, 0.0f, 0.0f);
        Attenuation.Constant = 1.0f;
        Attenuation.Linear = 0.0f;
        Attenuation.Exp = 0.0f;
    }
};
struct SpotLight : public PointLight
{
    vec3 Direction;
    float Cutoff;

    SpotLight()
    {
        Direction = vec3(0.0f, 0.0f, 0.0f);
        Cutoff = 0.0f;
    }
};

static const char* pVS = "                                                      \n\
    #version 330                                                                   \n\
    layout (location = 0) in vec3 pos;                                             \n\
    layout (location = 1) in vec2 tex;                                             \n\
    layout (location = 2) in vec3 norm;                                            \n\
    uniform mat4 gWVP;                                                             \n\
    uniform mat4 gWorld;                                                           \n\
    out vec3 pos0;                                                                 \n\
    out vec2 tex0;                                                                 \n\
    out vec3 norm0;                                                                \n\
    void main()                                                                    \n\
    {                                                                              \n\
        gl_Position = gWVP * vec4(pos, 1.0);                                       \n\
        tex0 = tex;                                                                \n\
        norm0 = (gWorld * vec4(norm, 0.0)).xyz;                                    \n\
        pos0 = (gWorld * vec4(pos, 1.0)).xyz;                                      \n\
    }";

static const char* pFS = "                                                                     \n\
    #version 330                                                                                \n\
const int MAX_POINT_LIGHTS = 3;                                                                 \n\
const int MAX_SPOT_LIGHTS = 2;                                                                  \n\
    in vec2 tex0;                                                                               \n\
    in vec3 norm0;                                                                              \n\
    in vec3 pos0;                                                                               \n\
    out vec4 fragcolor;                                                                         \n\
struct BaseLight                                                                                \n\
{                                                                                               \n\
    vec3 Color;                                                                                 \n\
    float AmbientIntensity;                                                                     \n\
    float DiffuseIntensity;                                                                     \n\
};                                                                                              \n\
struct DirectionalLight                                                                         \n\
{                                                                                               \n\
    BaseLight Base;                                                                             \n\
    vec3 Direction;                                                                             \n\
};                                                                                              \n\
struct Attenuation                                                                              \n\
{                                                                                               \n\
    float Constant;                                                                             \n\
    float Linear;                                                                               \n\
    float Exp;                                                                                  \n\
};                                                                                              \n\
struct PointLight                                                                               \n\
{                                                                                               \n\
    BaseLight Base;                                                                             \n\
    vec3 Position;                                                                              \n\
    Attenuation Atten;                                                                          \n\
};                                                                                              \n\
struct SpotLight                                                                                \n\
{                                                                                               \n\
    PointLight Base;                                                                            \n\
    vec3 Direction;                                                                             \n\
    float Cutoff;                                                                               \n\
};                                                                                              \n\
uniform int gNumPointLights;                                                                    \n\
uniform int gNumSpotLights;                                                                     \n\
uniform DirectionalLight gDirectionalLight;                                                     \n\
uniform PointLight gPointLights[MAX_POINT_LIGHTS];                                              \n\
uniform SpotLight gSpotLights[MAX_SPOT_LIGHTS];                                                 \n\
    uniform sampler2D gSampler;                                                                 \n\
    uniform vec3 gEyeWorldPos;                                                                  \n\
    uniform float gMatSpecularIntensity;                                                        \n\
    uniform float gSpecularPower;                                                               \n\
    vec4 CalcLightInternal(BaseLight Light, vec3 LightDirection, vec3 Normal){                  \n\
        vec4 AmbientColor = vec4(Light.Color, 1.0f) * Light.AmbientIntensity;                   \n\
        float DiffuseFactor = dot(Normal, -LightDirection);                                     \n\
                                                                                                \n\
        vec4 DiffuseColor  = vec4(0, 0, 0, 0);                                                  \n\
        vec4 SpecularColor = vec4(0, 0, 0, 0);                                                  \n\
                                                                                                \n\
        if (DiffuseFactor > 0) {                                                                \n\
            DiffuseColor = vec4(Light.Color, 1.0f) * Light.DiffuseIntensity * DiffuseFactor;    \n\
                                                                                                \n\
            vec3 VertexToEye = normalize(gEyeWorldPos - pos0);                                  \n\
            vec3 LightReflect = normalize(reflect(LightDirection, Normal));                     \n\
            float SpecularFactor = dot(VertexToEye, LightReflect);                              \n\
            SpecularFactor = pow(SpecularFactor, gSpecularPower);                               \n\
            if (SpecularFactor > 0) {                                                           \n\
                SpecularColor = vec4(Light.Color, 1.0f) *                                       \n\
                                gMatSpecularIntensity * SpecularFactor;                         \n\
            }                                                                                   \n\
        }                                                                                       \n\
        return (AmbientColor + DiffuseColor + SpecularColor);                                   \n\
    }                                                                                           \n\
                                                                                                \n\
    vec4 CalcDirectionalLight(vec3 Normal)                                                      \n\
    {                                                                                           \n\
        return CalcLightInternal(gDirectionalLight.Base, gDirectionalLight.Direction, Normal);  \n\
    }                                                                                           \n\
    vec4 CalcPointLight(PointLight l, vec3 Normal)                                              \n\
    {                                                                                           \n\
        vec3 LightDirection = pos0 - l.Position;                                                \n\
        float Distance = length(LightDirection);                                                \n\
        LightDirection = normalize(LightDirection);                                             \n\
                                                                                                \n\
        vec4 Color = CalcLightInternal(l.Base, LightDirection, Normal);                         \n\
        float Attenuation =  l.Atten.Constant +                                                 \n\
                             l.Atten.Linear * Distance +                                        \n\
                             l.Atten.Exp * Distance * Distance;                                 \n\
                                                                                                \n\
        return Color / Attenuation;                                                             \n\
    }                                                                                           \n\                                                                                            \n\
    vec4 CalcSpotLight(SpotLight l, vec3 Normal)                                                \n\
    {                                                                                           \n\
        vec3 LightToPixel = normalize(pos0 - l.Base.Position);                                  \n\
        float SpotFactor = dot(LightToPixel, l.Direction);                                      \n\
                                                                                                \n\
        if (SpotFactor > l.Cutoff) {                                                            \n\
            vec4 Color = CalcPointLight(l.Base, Normal);                                        \n\
            return Color * (1.0 - (1.0 - SpotFactor) * 1.0/(1.0 - l.Cutoff));                   \n\
        }                                                                                       \n\
        else return vec4(0,0,0,0);                                                              \n\
}                                                                                               \n\
    void main()                                                                                 \n\
    {                                                                                           \n\
        vec3 Normal = normalize(norm0);                                                         \n\
        vec4 TotalLight = CalcDirectionalLight(Normal);                                         \n\
                                                                                                \n\
        for (int i = 0 ; i < gNumPointLights ; i++) {                                           \n\
            TotalLight += CalcPointLight(gPointLights[i], Normal);                              \n\
        }                                                                                       \n\
        for (int i = 0 ; i < gNumSpotLights ; i++) {                                            \n\
            TotalLight += CalcSpotLight(gSpotLights[i], Normal);                                \n\
        }                                                                                       \n\
        fragcolor = texture2D(gSampler, tex0.xy) * TotalLight;                                  \n\
    }";
class Pipeline {
    struct PerspectiveProjInfo 
    {
        float FOV;
        float Width;
        float Height;
        float zNear;
        float zFar;
    };

    struct 
    {
        vec3 Pos;
        vec3 Target;
        vec3 Up;
    } m_camera;

    const int x = 0, y = 1, z = 2;

    vec3
        vScale{ 1.0f, 1.0f, 1.0f },
        vRotate{ 0.0f, 0.0f, 0.0f },
        vTranslation{ 0.0f, 0.0f, 0.0f };

    mat4 mTransformation;

    PerspectiveProjInfo perspectiveProjInfo{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    mat4 InitScaleTransform(float x, float y, float z) 
    {
         return mat4 {
            {x, 0.0f, 0.0f, 0.0f},
            {0.0f, y, 0.0f, 0.0f},
            {0.0f, 0.0f, z, 0.0f},
            {0.0f, 0.0f, 0.0f, 1.0f}
         };
    }

    mat4 InitRotateTransform(float _x, float _y, float _z) {
         glm::mat4 xm, ym, zm;

         float x = ToRadian(_x);
         float y = ToRadian(_y);
         float z = ToRadian(_z);

         xm = {
             {1.0f, 0.0f, 0.0f, 0.0f},
             {0.0f, cosf(x), -sinf(x), 0.0f},
             {0.0f, sinf(x), cosf(x), 0.0f},
             {0.0f, 0.0f, 0.0f, 1.0f},
         };

         ym = {
             {cosf(y), 0.0f, -sinf(y), 0.0f},
             {0.0f, 1.0f, 0.0f, 0.0f},
             {sinf(y), 0.0f, cosf(y), 0.0f},
             {0.0f, 0.0f, 0.0f, 1.0f},
         };

         zm = {
             {cosf(z), -sinf(z), 0.0f, 0.0f},
             {sinf(z), cosf(z), 0.0f, 0.0f},
             {0.0f, 0.0f, 1.0f, 0.0f},
             {0.0f, 0.0f, 0.0f, 1.0f},
         };

         return xm * ym * zm;
    }

    mat4 InitTranslationTransform(float x, float y, float z) {
         return glm::mat4{
             {1.0f, 0.0f, 0.0f, x},
             {0.0f, 1.0f, 0.0f, y},
             {0.0f, 0.0f, 1.0f, z},
             {0.0f, 0.0f, 0.0f, 1.0f},
         };
    }

    mat4 InitPerspectiveProj(float FOV, float Width, float Height, float zNear, float zFar) {
         const float ar = Width / Height;
         const float zRange = zNear - zFar;
         const float tanHalfFOV = tanf(ToRadian(FOV / 2.0));

         return mat4{
             {1.0f / (ar * tanHalfFOV), 0.0f, 0.0f, 0.0f},
             {0.0f, 1.0f / tanHalfFOV, 0.0f, 0.0f},
             {0.0f, 0.0f, (-zNear - zFar) / zRange, (2.0f * zFar * zNear) / zRange},
             {0.0f, 0.0f, 1.0f, 0.0f}
         };
    }

    mat4 InitCameraTransform(glm::vec3 Target, glm::vec3 Up) {
         vec3 N = glm::vec3(Target);
         VectorsMath::Normalize(N);
         vec3 _U = glm::vec3(Up);
         VectorsMath::Normalize(_U);
         vec3 U = VectorsMath::Cross(_U, Target);
         vec3 V = VectorsMath::Cross(N, U);

         return glm::mat4{
             {U[x], U[y], U[z], 0.0f},
             {V[x], V[y], V[z], 0.0f},
             {N[x], N[y], N[z], 0.0f},
             {0.0f, 0.0f, 0.0f, 1.0f},
         };
    }

public:
    void Scale(float x, float y, float z) {
         vScale = { x, y, z };
    }

    void WorldPos(float x, float y, float z) {
         vTranslation = { x, y, z };
    }

    void Rotate(float x, float y, float z) {
         vRotate = { x, y, z };
    }

    void SetPerspectiveProj(float FOV, float Width, float Height, float zNear, float zFar) {
         perspectiveProjInfo = { FOV, Width, Height, zNear, zFar };
    }

    void SetCamera(vec3 pos, vec3 target, vec3 up) {
         m_camera = { pos, target, up };
    }

    mat4 GetTrans() {
         mat4 ScaleTrans, RotateTrans, TranslationTrans, CameraTranslationTrans, CameraRotateTrans, PersProjTrans;

         ScaleTrans = InitScaleTransform(vScale[x], vScale[y], vScale[z]);
         RotateTrans = InitRotateTransform(vRotate[x], vRotate[y], vRotate[z]);
         TranslationTrans = InitTranslationTransform(vTranslation[x], vTranslation[y], vTranslation[z]);
         CameraTranslationTrans = InitTranslationTransform(-m_camera.Pos[x], -m_camera.Pos[y], -m_camera.Pos[z]);
         CameraRotateTrans = InitCameraTransform(m_camera.Target, m_camera.Up);
         PersProjTrans = InitPerspectiveProj(perspectiveProjInfo.FOV, perspectiveProjInfo.Width, perspectiveProjInfo.Height, perspectiveProjInfo.zNear, perspectiveProjInfo.zFar);

         //return PersProjTrans * CameraRotateTrans * CameraTranslationTrans * TranslationTrans * RotateTrans * ScaleTrans; 
         //return PersProjTrans * CameraRotateTrans * CameraTranslationTrans * TranslationTrans *
         //RotateTrans* ScaleTrans;
         return ScaleTrans * RotateTrans * TranslationTrans;
    }
};

class Texture {
private:
    string filename;
    GLenum textarget;
    GLuint texobj;
    Image* image;
    Blob blob;
public:
    Texture(GLenum target, const string& name) {
        textarget = target;
        filename = name;
        image = nullptr;
    };
    bool load() {
        try {
            image = new Image(filename);
            image->write(&blob, "RGBA");
        }
        catch (Error& Error) {
            cerr << "Error loading texture '" << filename << "': " << Error.what() << endl;
            return false;
        }

        glGenTextures(1, &texobj);
        glBindTexture(textarget, texobj);
        glTexImage2D(textarget, 0, GL_RGB, image->columns(), image->rows(), -0.5, GL_RGBA, GL_UNSIGNED_BYTE, blob.data());
        glTexParameterf(textarget, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameterf(textarget, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        return true;
    };
    void bind(GLenum unit) {
        glActiveTexture(unit);
        glBindTexture(textarget, texobj);
    };

};

class Technique {
private:
    GLuint program;
    typedef list<GLuint> shaderlist;
    shaderlist shaders;
protected:
    bool AddShader(GLenum shadertype, const char* shadertext) {
        GLuint shader = glCreateShader(shadertype);

        const GLchar* ShaderSource[1];
        ShaderSource[0] = shadertext;
        glShaderSource(shader, 1, ShaderSource, nullptr);

        glCompileShader(shader);

        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success)
        {
            glGetShaderInfoLog(shader, sizeof(InfoLog), nullptr, InfoLog);
            cerr << "Error compiling shader: " << InfoLog << endl;
            return false;
        }
        glAttachShader(program, shader);
        return true;
    };
    bool Link() {
        glLinkProgram(program);

        glGetProgramiv(program, GL_LINK_STATUS, &success);
        if (success == 0) {
            glGetProgramInfoLog(program, sizeof(InfoLog), nullptr, InfoLog);
            cerr << "Error linking shader program: " << InfoLog << endl;
            return false;
        }
        /*
        glValidateProgram(m_shaderProg);
        glGetProgramiv(m_shaderProg, GL_VALIDATE_STATUS, &success);
        if (success == 0) {
            glGetProgramInfoLog(m_shaderProg, sizeof(InfoLog), nullptr, InfoLog);
            cerr << "Invalid shader program: " << InfoLog << endl;
            return false;
        }*/

        for (shaderlist::iterator it = shaders.begin(); it != shaders.end(); it++) {
            glDeleteShader(*it);
        }

        shaders.clear();

        return true;
    };
    GLint GetUniformLocation(const char* name) {
        GLint Location = glGetUniformLocation(program, name);

        if (Location == 0xFFFFFFFF) {
            cerr << "Warning! Unable to get the location of uniform " << name << endl;
        }
        return Location;
    };
public:
    Technique() { program = 0; };
    ~Technique() {
        for (shaderlist::iterator it = shaders.begin(); it != shaders.end(); it++) {
            glDeleteShader(*it);
        }

        if (program != 0) {
            glDeleteProgram(program);
            program = 0;
        }
    };
    virtual bool Init() {
        program = glCreateProgram();

        if (program == 0) {
            cerr << "Error creating shader program: " << InfoLog << endl;
            return false;
        }

        return true;
    };
    void Enable() { glUseProgram(program); };
};

class LightingTechnique : public Technique {
private:
    GLuint gWVPLocation;
    GLuint gWorldLocation;
    GLuint samplerLocation;

    GLuint dirLightColor;
    GLuint dirLightAmbientIntensity;
    GLuint dirLightDirection;
    GLuint dirLightDiffuseIntensity;

    GLuint eyeWorldPosition;
    GLuint matSpecularIntensityLocation;
    GLuint matSpecularPowerLocation;

    GLuint numPointLightsLocation;
    GLuint numSpotLightsLocation;

    struct {
        GLuint Color;
        GLuint AmbientIntensity;
        GLuint DiffuseIntensity;
        GLuint Position;
        struct {
            GLuint Constant;
            GLuint Linear;
            GLuint Exp;
        } Atten;
    } pointLights[MAX_POINT_LIGHTS];

    struct {
        GLuint Color;
        GLuint AmbientIntensity;
        GLuint DiffuseIntensity;
        GLuint Position;
        GLuint Direction;
        GLuint Cutoff;
        struct {
            GLuint Constant;
            GLuint Linear;
            GLuint Exp;
        } Atten;
    } spotLights[MAX_SPOT_LIGHTS];
public:
    LightingTechnique() {};

    virtual bool Init() {
        if (!Technique::Init()) return false;
        if (!AddShader(GL_VERTEX_SHADER, pVS)) return false;
        if (!AddShader(GL_FRAGMENT_SHADER, pFS)) return false;
        if (!Link())  return false;

        gWVPLocation = GetUniformLocation("gWVP");
        gWorldLocation = GetUniformLocation("gWorld");
        samplerLocation = GetUniformLocation("gSampler");

        dirLightColor = GetUniformLocation("gDirectionalLight.Base.Color");
        dirLightAmbientIntensity = GetUniformLocation("gDirectionalLight.Base.AmbientIntensity");
        dirLightDirection = GetUniformLocation("gDirectionalLight.Direction");
        dirLightDiffuseIntensity = GetUniformLocation("gDirectionalLight.Base.DiffuseIntensity");

        eyeWorldPosition = GetUniformLocation("gEyeWorldPos");
        matSpecularIntensityLocation = GetUniformLocation("gMatSpecularIntensity");
        matSpecularPowerLocation = GetUniformLocation("gSpecularPower");

        numPointLightsLocation = GetUniformLocation("gNumPointLights");
        numSpotLightsLocation = GetUniformLocation("gNumSpotLights");

        for (unsigned int i = 0; i < MAX_POINT_LIGHTS; i++) {
            char Name[128];
            memset(Name, 0, sizeof(Name));
            snprintf(Name, sizeof(Name), "gPointLights[%d].Base.Color", i);
            pointLights[i].Color = GetUniformLocation(Name);

            snprintf(Name, sizeof(Name), "gPointLights[%d].Base.AmbientIntensity", i);
            pointLights[i].AmbientIntensity = GetUniformLocation(Name);

            snprintf(Name, sizeof(Name), "gPointLights[%d].Position", i);
            pointLights[i].Position = GetUniformLocation(Name);

            snprintf(Name, sizeof(Name), "gPointLights[%d].Base.DiffuseIntensity", i);
            pointLights[i].DiffuseIntensity = GetUniformLocation(Name);

            snprintf(Name, sizeof(Name), "gPointLights[%d].Atten.Constant", i);
            pointLights[i].Atten.Constant = GetUniformLocation(Name);

            snprintf(Name, sizeof(Name), "gPointLights[%d].Atten.Linear", i);
            pointLights[i].Atten.Linear = GetUniformLocation(Name);

            snprintf(Name, sizeof(Name), "gPointLights[%d].Atten.Exp", i);
            pointLights[i].Atten.Exp = GetUniformLocation(Name);

            if (pointLights[i].Color == 0xFFFFFFFF ||
                pointLights[i].AmbientIntensity == 0xFFFFFFFF ||
                pointLights[i].Position == 0xFFFFFFFF ||
                pointLights[i].DiffuseIntensity == 0xFFFFFFFF ||
                pointLights[i].Atten.Constant == 0xFFFFFFFF ||
                pointLights[i].Atten.Linear == 0xFFFFFFFF ||
                pointLights[i].Atten.Exp == 0xFFFFFFFF) return false;
        }

        for (unsigned int i = 0; i < MAX_SPOT_LIGHTS; i++) {
            char Name[128];
            memset(Name, 0, sizeof(Name));
            snprintf(Name, sizeof(Name), "gSpotLights[%d].Base.Base.Color", i);
            spotLights[i].Color = GetUniformLocation(Name);

            snprintf(Name, sizeof(Name), "gSpotLights[%d].Base.Base.AmbientIntensity", i);
            spotLights[i].AmbientIntensity = GetUniformLocation(Name);

            snprintf(Name, sizeof(Name), "gSpotLights[%d].Base.Position", i);
            spotLights[i].Position = GetUniformLocation(Name);

            snprintf(Name, sizeof(Name), "gSpotLights[%d].Direction", i);
            spotLights[i].Direction = GetUniformLocation(Name);

            snprintf(Name, sizeof(Name), "gSpotLights[%d].Cutoff", i);
            spotLights[i].Cutoff = GetUniformLocation(Name);

            snprintf(Name, sizeof(Name), "gSpotLights[%d].Base.Base.DiffuseIntensity", i);
            spotLights[i].DiffuseIntensity = GetUniformLocation(Name);

            snprintf(Name, sizeof(Name), "gSpotLights[%d].Base.Atten.Constant", i);
            spotLights[i].Atten.Constant = GetUniformLocation(Name);

            snprintf(Name, sizeof(Name), "gSpotLights[%d].Base.Atten.Linear", i);
            spotLights[i].Atten.Linear = GetUniformLocation(Name);

            snprintf(Name, sizeof(Name), "gSpotLights[%d].Base.Atten.Exp", i);
            spotLights[i].Atten.Exp = GetUniformLocation(Name);

            if (spotLights[i].Color == 0xFFFFFFFF ||
                spotLights[i].AmbientIntensity == 0xFFFFFFFF ||
                spotLights[i].Position == 0xFFFFFFFF ||
                spotLights[i].Direction == 0xFFFFFFFF ||
                spotLights[i].Cutoff == 0xFFFFFFFF ||
                spotLights[i].DiffuseIntensity == 0xFFFFFFFF ||
                spotLights[i].Atten.Constant == 0xFFFFFFFF ||
                spotLights[i].Atten.Linear == 0xFFFFFFFF ||
                spotLights[i].Atten.Exp == 0xFFFFFFFF) return false;
        }

        if (dirLightAmbientIntensity == 0xFFFFFFFF ||
            gWorldLocation == 0xFFFFFFFF ||
            samplerLocation == 0xFFFFFFFF ||
            dirLightColor == 0xFFFFFFFF ||
            dirLightDirection == 0xFFFFFFFF ||
            dirLightDiffuseIntensity == 0xFFFFFFFF ||
            eyeWorldPosition == 0xFFFFFFFF ||
            matSpecularIntensityLocation == 0xFFFFFFFF ||
            matSpecularPowerLocation == 0xFFFFFFFF ||
            numPointLightsLocation == 0xFFFFFFFF ||
            numSpotLightsLocation == 0xFFFFFFFF) return false;

        return true;
    };

    void SetgWVP(const mat4x4* gWorld) {
        glUniformMatrix4fv(gWVPLocation, 1, GL_TRUE, (const GLfloat*)gWorld);
    };

    void SetWorld(const mat4x4* World)
    {
        glUniformMatrix4fv(gWorldLocation, 1, GL_TRUE, (const GLfloat*)World);
    }

    void SetTextureUnit(unsigned int unit) {
        glUniform1i(samplerLocation, unit);
    };

    void SetMatSpecularIntensity(float Intensity)
    {
        glUniform1f(matSpecularIntensityLocation, Intensity);
    }

    void SetMatSpecularPower(float Power)
    {
        glUniform1f(matSpecularPowerLocation, Power);
    }

    void SetEyeWorldPos(const vec3& EyeWorldPos)
    {
        glUniform3f(eyeWorldPosition, EyeWorldPos.x, EyeWorldPos.y, EyeWorldPos.z);
    }

    void SetDirectionalLight(const DirectionalLight& Light) {
        glUniform3f(dirLightColor, Light.Color.x, Light.Color.y, Light.Color.z);
        glUniform1f(dirLightAmbientIntensity, Light.AmbientIntensity);
        vec3 Direction = Light.Direction;
        normalize(Direction);
        glUniform3f(dirLightDirection, Direction.x, Direction.y, Direction.z);
        glUniform1f(dirLightDiffuseIntensity, Light.DiffuseIntensity);
    };

    void SetPointLights(unsigned int NumLights, const PointLight* Lights)
    {
        glUniform1i(numPointLightsLocation, NumLights);

        for (unsigned int i = 0; i < NumLights; i++) {
            glUniform3f(pointLights[i].Color, Lights[i].Color.x, Lights[i].Color.y, Lights[i].Color.z);
            glUniform1f(pointLights[i].AmbientIntensity, Lights[i].AmbientIntensity);
            glUniform1f(pointLights[i].DiffuseIntensity, Lights[i].DiffuseIntensity);
            glUniform3f(pointLights[i].Position, Lights[i].Position.x, Lights[i].Position.y, Lights[i].Position.z);
            glUniform1f(pointLights[i].Atten.Constant, Lights[i].Attenuation.Constant);
            glUniform1f(pointLights[i].Atten.Linear, Lights[i].Attenuation.Linear);
            glUniform1f(pointLights[i].Atten.Exp, Lights[i].Attenuation.Exp);
        }
    }

    void SetSpotLights(unsigned int NumLights, const SpotLight* Lights)
    {
        glUniform1i(numSpotLightsLocation, NumLights);

        for (unsigned int i = 0; i < NumLights; i++) {
            glUniform3f(spotLights[i].Color, Lights[i].Color.x, Lights[i].Color.y, Lights[i].Color.z);
            glUniform1f(spotLights[i].AmbientIntensity, Lights[i].AmbientIntensity);
            glUniform1f(spotLights[i].DiffuseIntensity, Lights[i].DiffuseIntensity);
            glUniform3f(spotLights[i].Position, Lights[i].Position.x, Lights[i].Position.y, Lights[i].Position.z);
            vec3 Direction = Lights[i].Direction;
            normalize(Direction);
            glUniform3f(spotLights[i].Direction, Direction.x, Direction.y, Direction.z);
            glUniform1f(spotLights[i].Cutoff, cosf(radians(Lights[i].Cutoff)));
            glUniform1f(spotLights[i].Atten.Constant, Lights[i].Attenuation.Constant);
            glUniform1f(spotLights[i].Atten.Linear, Lights[i].Attenuation.Linear);
            glUniform1f(spotLights[i].Atten.Exp, Lights[i].Attenuation.Exp);
        }
    }
};

class ICallbacks
{
public:
    virtual void KeyboardCB(unsigned char Key, int x, int y) = 0;
    virtual void RenderSceneCB() = 0;
    virtual void IdleCB() = 0;
};

void GLUTBackendInit(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    InitializeMagick(*argv);
};

bool GLUTBackendCreateWindow(unsigned int Width, unsigned int Height, const char* name) {
    glutInitWindowSize(Width, Height);
    glutInitWindowPosition(100, 100);
    glutCreateWindow(name);
    GLenum res = glewInit();
    if (res != GLEW_OK) {
        cerr << "Error: " << glewGetErrorString(res) << endl;
        return false;
    }
    return true;
};

ICallbacks* callbacks = nullptr;

void RenderScene() {
    callbacks->RenderSceneCB();
}

void Idle() {
    callbacks->IdleCB();
}

void Keyboard(unsigned char Key, int x, int y) {
    callbacks->KeyboardCB(Key, x, y);
}

void cb() {
    glutDisplayFunc(RenderScene);
    glutIdleFunc(Idle);
    glutKeyboardFunc(Keyboard);
}

void GLUTBackendRun(ICallbacks* p) {
    if (!p) {
        fprintf(stderr, "%s : callbacks not specified!\n", __FUNCTION__);
        return;
    }

    callbacks = p;
    cb();
    glutMainLoop();
};

class Main : public ICallbacks
{
private:
    GLuint VBO;
    GLuint IBO;
    LightingTechnique* eff;

    Texture* tex;
    DirectionalLight dirLight;
    void genbuffers() {
        Vertex Vertices[4] = {
          Vertex(vec3(-0.2, -0.2, 0),vec2(0,0)),
          Vertex(vec3(0.3, -0.2, 0.5),vec2(0.5,0)),
          Vertex(vec3(0.3, -0.2, -0.5),vec2(1,0)),
          Vertex(vec3(0, 0.4, 0),vec2(0.5,1)),
        };


        unsigned int Indices[] = { 0, 3, 1,
                                   1, 3, 2,
                                   2, 3, 0,
                                   0, 2, 1 };

        calcnorms(Indices, 12, Vertices, 4);

        glGenBuffers(1, &VBO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(Vertices), Vertices, GL_STATIC_DRAW);


        glGenBuffers(1, &IBO);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(Indices), Indices, GL_STATIC_DRAW);
    }
    void calcnorms(const unsigned int* indices, unsigned int indcount, Vertex* vertices, unsigned int vertcount) {
        for (unsigned int i = 0; i < indcount; i += 3) {
            unsigned int Index0 = indices[i];
            unsigned int Index1 = indices[i + 1];
            unsigned int Index2 = indices[i + 2];
            vec3 v1 = vertices[Index1].m_pos - vertices[Index0].m_pos;
            vec3 v2 = vertices[Index2].m_pos - vertices[Index0].m_pos;
            vec3 norm = cross(v1, v2);
            VectorsMath::Normalize(norm);

            vertices[Index0].m_norm += norm;
            vertices[Index1].m_norm += norm;
            vertices[Index2].m_norm += norm;
        }

        for (unsigned int i = 0; i < vertcount; i++) {
            VectorsMath::Normalize(vertices[i].m_norm);
        }
    }
public:
    Main()
    {
        tex = nullptr;
        eff = nullptr;
        dirLight.Color = vec3(1.0f, 1.0f, 1.0f);
        dirLight.AmbientIntensity = 0.5;
        dirLight.DiffuseIntensity = 0.9f;
        dirLight.Direction = vec3(0.0f, 0.0, -1.0);
    }
    ~Main() {
        delete eff;
        delete tex;
    };
    bool Init()
    {
        genbuffers();
        eff = new LightingTechnique();
        if (!eff->Init())
        {
            return false;
        }
        eff->Enable();
        eff->SetTextureUnit(0);

        tex = new Texture(GL_TEXTURE_2D, "Test.png");

        if (!tex->load()) {
            return false;
        }

        return true;
    }

    void Run()
    {
        GLUTBackendRun(this);
    }

    virtual void RenderSceneCB()
    {
        glClearColor(0.0f, 0.0, 0.0, 0);
        glClear(GL_COLOR_BUFFER_BIT);

        scale += 0.1f;


        Pipeline p;
        //p.Scale(1, 1, 1);
        p.WorldPos(0.0f, 0.0f, 0.0f);
        p.Rotate(0, scale, 0);
        //p.SetPerspectiveProj(60.0f, WINDOW_WIDTH, WINDOW_HEIGHT, 1.0f, 100.0f);
        vec3 CamPos(0.0f, 0.0f, 0.0f);
        vec3 CamTarget(0.0f, 0.0f, 2.0f);
        vec3 CamUp(0.0f, 1.0f, 0.0f);
        //p.SetCamera(CamPos, CamTarget, CamUp);

        const mat4x4 WorldTransformation = p.GetTrans();
        eff->SetgWVP(&WorldTransformation);
        eff->SetWorld(&WorldTransformation);
        eff->SetDirectionalLight(dirLight);

        eff->SetEyeWorldPos(CamPos);
        eff->SetMatSpecularIntensity(0.5f);
        eff->SetMatSpecularPower(32);

        SpotLight sl[2];
        sl[0].DiffuseIntensity = 50;
        sl[0].Color = vec3(0, 1, 0);
        sl[0].Position = vec3(0, 0, -0.0f);
        sl[0].Direction = vec3(sinf(scale), 0.0f, cosf(scale));
        sl[0].Attenuation.Linear = 0.1f;
        sl[0].Cutoff = 20;

        sl[1].DiffuseIntensity = 1;
        sl[1].Color = vec3(0, 0, 1);
        sl[1].Position = -CamPos;
        sl[1].Direction = -CamTarget;
        sl[1].Attenuation.Linear = 0.1f;
        sl[1].Cutoff = 10;

        eff->SetSpotLights(2, sl);

        PointLight pl[3];
        pl[0].DiffuseIntensity = 0.5;
        pl[0].Color = vec3(1.0f, 0.0f, 0.0f);
        pl[0].Position = vec3(sinf(scale) * 10, 1.0f, cosf(scale) * 10);
        pl[0].Attenuation.Linear = 0.1f;

        pl[1].DiffuseIntensity = 0.5;
        pl[1].Color = vec3(0.0f, 1.0f, 0.0f);
        pl[1].Position = vec3(sinf(scale + 2.1f) * 10, 1.0f, cosf(scale + 2.1f) * 10);
        pl[1].Attenuation.Linear = 0.1f;

        pl[2].DiffuseIntensity = 0.5;
        pl[2].Color = vec3(0.0f, 0.0f, 1.0f);
        pl[2].Position = vec3(sinf(scale + 4.2f) * 10, 1.0f, cosf(scale + 4.2f) * 10);
        pl[2].Attenuation.Linear = 0.1f;

        eff->SetPointLights(3, pl);

        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glEnableVertexAttribArray(2);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), 0);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (const GLvoid*)12);
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (const GLvoid*)20);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IBO);
        tex->bind(GL_TEXTURE0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

        glDrawElements(GL_TRIANGLES, 12, GL_UNSIGNED_INT, 0);

        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);
        glDisableVertexAttribArray(2);

        glutPostRedisplay();

        glutSwapBuffers();
    }
    virtual void IdleCB()
    {
        RenderSceneCB();
    }
    virtual void KeyboardCB(unsigned char Key, int x, int y)
    {
        switch (Key) {
        case 'q':
            glutLeaveMainLoop();
            break;

        case 'a':
            dirLight.AmbientIntensity += 0.05f;
            break;

        case 's':
            dirLight.AmbientIntensity -= 0.05f;
            break;

        case 'z':
            dirLight.DiffuseIntensity += 0.05f;
            break;

        case 'x':
            dirLight.DiffuseIntensity -= 0.05f;
            break;
        }
    }
};

int main(int argc, char** argv)
{
    GLUTBackendInit(argc, argv);

    if (!GLUTBackendCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Third Lab CG")) {
        return 1;
    }

    Main* pApp = new Main();

    if (!pApp->Init()) return 1;

    pApp->Run();

    delete pApp;

    return 0;
}