import pycuda.driver as cuda
import numpy as np
from pycuda.autoinit import context
from pycuda.compiler import SourceModule
import random

def rand_poly_string(degree, var = 'u'):
    s = ""
    for i in range(degree + 1):
        s = s + "{}*".format(round(random.uniform(-1/(i+1), 1/(i+1)), 3));
        for j in range(i):
            s += "{}*".format(var)
        s = s[:-1] + '+'
    return s[:-1]

def rand_multi_poly_string(degree):
    return rand_poly_string(degree, 'u') + '*' + rand_poly_string(degree, 'v') 

def rand_interpolate_poly(degree):
    s2 = rand_poly_string(degree, 'u') + '*' + rand_poly_string(degree, 'v')
    s1 = rand_poly_string(degree, 'u') + '*' + rand_poly_string(degree, 'v') 
    return '({})*t + (1-t)*({})'.format(s1, s2)

def rand_multi_fourier_series(N = 4):
    s1 = rand_fourier_series(var = 'v', N = N)
    s2 = rand_fourier_series(var = 'u', N = N)
    return '({})+({})'.format(s1, s2)

def interpolate(s1, s2):
    return '({})*t + (1-t)*{}'.format(s1, s2)

def rand_interpolate_fourier_series(N = 4):
    s1 = rand_multi_fourier_series(N)
    s2 = rand_multi_fourier_series(N)
    s2 = '0'
    return '({})*t + (1-t)*({})'.format(s1, s2)

def rand_fourier_series(var = 'v',N = 10):
    N = random.randint(1, N)
    s = ''
    for n in range(1, N+1):
        an = round(random.uniform(-0.25, 0.25), 3)
        bn = round(random.uniform(-0.25, 0.25), 3)
        s = s + '({})*cosf(({})*({}))+'.format(an, var, n)#+{}*sin({}*{})+'.format(an, var,n, bn,var, n)
    return s[:-1]


deg = 5

class GPUPatch():
    def __init__(self, bounds = (-3.14, 3.14, -3.14, 3.14), updates = True, square=True, time_bounds = (0, 1)):
        self.updates = True
        du, dv = (0.03, 0.03)
        uMin, uMax, vMin, vMax = bounds 
        self.is_square = square
        self.curvature_type = 0
        self.animate = False
        self._time = time_bounds[0]
        self.ticks = int(self._time/0.01)
        self.time_bounds = time_bounds
        self.time_direction = 1
        self.is_paused = False
        self.bounds = bounds

        self.gpu_mod = SourceModule("""
            
            __constant__ float uMin = """ + str(uMin) + """;

            __constant__ float uMax = """ + str(uMax) + """;

            __constant__ float vMin = """ + str(vMin) + """;

            __constant__ float vMax = """ + str(vMax) + """;

            __constant__ float du = """ + str(du) + """;

            __constant__ float dv = """ + str(dv) + """;
            
            __constant__ float _d = 0.02;
            
            __device__ float3 uVel, vVel;

            __device__ float phi_x(float u, float v, float t) {
                //return (cos(u) + 2)*cos(v);
                //return ((cos(u) + 2)*cos(v) * t) + (1 - t)*u;
                //return sin(u)*cos(v);
                //return (2 + u*0.5*cos(v*0.5))*cos(v) * t + (1-t)*
                //return """ + interpolate(rand_multi_poly_string(deg), 'u') + """;
                return """ +  interpolate(rand_multi_fourier_series(deg), 'u')+ """;
             
            }

            __device__ float phi_y(float u, float v, float t) {
                //return (cos(u) + 2)*sin(v);
                //return ((cos(u) + 2)*sin(v));// * t) + (1 - t) *v;
                //return v;
                //return sin(u)*sin(v);
                //(2 + u*0.5*cos(v*0.5))*sin(v);
                //return cosh(v)*sin(u);
                //return v*t + (1 - t)* """ + rand_fourier_series('u',deg) + """;
                //return """ + interpolate(rand_multi_poly_string(deg), 'v') + """;
                return """ + interpolate(rand_multi_fourier_series(deg), 'v') + """;
            }

            __device__ float phi_z(float u, float v, float t) {
                //return sin(u);
                //return cos(u);
                //return u*0.5*sin(v*0.5) * t + (1-t);
                //return v;
                //return u*v;
                //return u*t + (1 - t)* """ + rand_fourier_series('1',deg) + """;
                //return """ + interpolate(rand_multi_poly_string(deg), '1') + """;
                return """ + interpolate(rand_multi_fourier_series(deg), '1') + """;
            }

            __device__ float3 """ + self.patch_function() + """

           
            __device__ float dot(float3 vec1, float3 vec2){
                return vec1.x*vec2.x + vec1.y*vec2.y + vec1.z * vec2.z;
            }

            __device__ float magnitude(float3 vec){
                return sqrt(dot(vec, vec));
            }
            
            __device__ float3 cross(float3 u, float3 v){
                return {u.y*v.z - u.z*v.y, u.z*v.x - u.x*v.z, u.x*v.y - u.y*v.x};
            }

            __device__ float3 normalized(float3 vec){
                float mag = magnitude(vec);
                return {vec.x / mag, vec.y / mag, vec.z / mag};
            }
           
            __device__ float3 subtract(float3 v, float3 w){
                return {v.x - w.x, v.y - w.y, v.z - w.z};
            }
            
            __device__ float3 add(float3 v, float3 w){
                return {v.x + w.x, v.y + w.y, v.z + w.z};
            }
            __device__ float3 multiply(float a, float3 v){
                return {a*v.x, a*v.y, a*v.z};

            }

            __device__ float3 phi(float u, float v, float t, bool animate){
                float3 phi_uv;
                 if(!animate){
                        phi_uv = phii(u, v, t);
                    }
                    else{
                        float3 b = {u, v, 0};
                        phi_uv = add(multiply(t, b),multiply((1-t),phii(u,v,0)));
                    }    
                return phi_uv;
            }

            __device__ float3 uVelocity(float u, float v, float t, bool animate) {
                
                float3 phi_uv = phi(u, v, t, animate);
                float3 phi_duv = phi(u + _d, v, t, animate);
                float dx = (phi_duv.x - phi_uv.x) / _d;
                float dy = (phi_duv.y - phi_uv.y) / _d;
                float dz = (phi_duv.z - phi_uv.z) / _d;
                return {dx, dy, dz};
           }
            __device__ float3 vVelocity(float u, float v, float t, bool animate) {
                float3 phi_uv = phi(u, v, t, animate);
                float3 phi_duv = phi(u, v + _d, t, animate);
                float dx = (phi_duv.x - phi_uv.x) / _d;
                float dy = (phi_duv.y - phi_uv.y) / _d;
                float dz = (phi_duv.z - phi_uv.z) / _d;
                return {dx, dy, dz};
            }
             
            __device__ float3 unitNormal(float u, float v, float t, bool animate){
                return normalized(cross(uVelocity(u, v, t, animate), vVelocity(u, v, t, animate)));
            }
 
            __device__ float3 shapeOperator(float u, float v, float t, float3 direction, bool animate){
               
                float3 normal = unitNormal(u, v, t, animate);//cross(uVel, vVel);
               
                direction = multiply(_d, direction);
                float2 pullback = {dot(uVel, direction), dot(vVel, direction)};
                
                uVel = uVelocity(u + pullback.x, v, t, animate);
                vVel = vVelocity(u, v + pullback.y, t, animate);

                float3 nextNormal = cross(uVel, vVel);
                return {-1*(nextNormal.x - normal.x)/_d, -1*(nextNormal.y - normal.y)/_d, -1*(nextNormal.z - normal.z)/_d};
            }

            __device__ float normalCurvature(float u, float v, float t, float3 direction, bool animate){
                return dot(direction, shapeOperator(u, v, t, direction, animate));
            }

            __device__ float meanCurvature(float u, float v, float t, bool animate){
                
                float3 uVel = uVelocity(u, v, t, animate);
                float3 vVel = vVelocity(u, v, t, animate);
                
                float E = dot(uVel, uVel);
                float G = dot(vVel, vVel);
                float F = dot(uVel, vVel);
                
                
                float3 shapeU = shapeOperator(u, v, t, uVel, animate);
                float3 shapeV = shapeOperator(u, v, t, vVel, animate);
                float L = dot(shapeU, uVel);
                float N = dot(shapeV, vVel);
                float M = dot(shapeV, uVel);
                
                return (G*L + E*N - 2*F*M)/ (2*(E*G - F*F));

            }
            __device__ float gaussianCurvature(float u, float v, float t, bool animate){
                
                float3 uVel = uVelocity(u, v, t, animate);
                float3 vVel = vVelocity(u, v, t, animate);
                
                float E = dot(uVel, uVel);
                float G = dot(vVel, vVel);
                float F = dot(uVel, vVel);
                
                float3 shapeU = shapeOperator(u, v, t, uVel, animate);
                float3 shapeV = shapeOperator(u, v, t, vVel, animate);
                
                float L = dot(shapeU, uVel);
                float N = dot(shapeV, vVel);
                float M = dot(shapeV, uVel);
                return ((L*N) - (M*M))/((E*G) - (F*F));

           }

            __global__ void compute_verts(float* out, float* time, bool square, bool animate){
                // let height = (uMax - uMin) // du, width = (vMax - vMin) // dv
                // Here, out is a flattened matrix M of dimension (height, width, 3) where
                // out[i*width + j*3 + k] = M[i, j, k]
                // and M[i, j] = phi(uMin + i*du, vMin + j*dv)
                // phi : R2 -> R3 is a smooth map determined by constant phi_func

                int width = (int) ((vMax - vMin) / dv);
                int height = (int) ((uMax - uMin) / du);
                float st = time[0];
                
                int i = threadIdx.x;
                
                for(int j = 0; j < width; ++j){
                    float up = uMin + (i * du);
                    float vp = vMin + (j * dv);
                    float u = up;
                    float v = vp;
                    if(!square){
                        u = up*cosf(vp);
                        v = up*sinf(vp);
                    }
                    float3 phi_uv = phi(u, v, st, animate);
                    out[((i*width + j)*3)] = phi_uv.x;
                    out[((i*width + j)*3 + 1)] = phi_uv.y;
                    out[((i*width + j)*3 + 2)] = phi_uv.z;
                    
                    /*
                    out[i + height*(j + width*0)] = phi_uv.x;
                    out[i + height*(j + width*1)] = phi_uv.y;
                    out[i + height*(j + width*2)] = phi_uv.z;
                    */
               }
            } 
            
            __global__ void compute_tris(float* out, float* verts){

                int width = (int) ((vMax - vMin) / dv);
                int i = threadIdx.x;

                for(int j = 0; j < width; ++j){
                     if(i == 0 || j == 0 || i >= width-3 || j >= width - 3){
                        continue;
                    }
                   
                    if(j % 2 == 0){
                        out[(i*width + j)*9] = verts[(i*width + j)*3];
                        out[(i*width + j)*9 + 1] = verts[(i*width +j)*3 + 1];
                        out[(i*width + j)*9 + 2] = verts[(i * width + j)*3 + 2];
                        
                        out[((i*width + j)*3 + 1)*3] = verts[(i*width + j+2)*3];
                        out[((i*width + j)*3 + 1)*3 + 1] = verts[(i*width + j+2)*3 + 1];
                        out[((i*width + j)*3 + 1)*3 + 2] = verts[(i*width + j+2)*3 + 2];
 
                        out[((i*width + j)*3 +2)*3] = verts[((i+2)*width + j)*3];
                        out[((i*width + j)*3 +2)*3 + 1] = verts[((i+2)*width +j)*3 + 1];
                        out[((i*width + j)*3 +2)*3 + 2] = verts[((i+2) * width + j)*3 + 2];
                    }
                    else{
                        out[(i*width + j)*9] = verts[(i*width + j+1)*3];
                        out[(i*width + j)*9 + 1] = verts[(i*width +j+1)*3 + 1];
                        out[(i*width + j)*9 + 2] = verts[(i * width + j+1)*3 + 2];
                        
                        out[((i*width + j)*3 + 2)*3] = verts[((i+2)*width + j-1)*3];
                        out[((i*width + j)*3 + 2)*3 + 1] = verts[((i+2)*width + j-1)*3 + 1];
                        out[((i*width + j)*3 + 2)*3 + 2] = verts[((i+2)*width + j-1)*3 + 2];
 
                        out[((i*width + j)*3 +1)*3] = verts[((i+2)*width + j+1)*3];
                        out[((i*width + j)*3 +1)*3 + 1] = verts[((i+2)*width +j+1)*3 + 1];
                        out[((i*width + j)*3 +1)*3 + 2] = verts[((i+2) * width + j+1)*3 + 2];
                    }
                }
            }
            __global__ void compute_tris_normals(float* out, float* time, bool animate){
                int width = (int) ((vMax - vMin) / dv);
                int i = threadIdx.x;
                
                float st = time[0];
                
                for(int j = 0; j < width; ++j){
                    if(i == 0 || j == 0 || i >= width-3 || j >= width - 3){
                        continue;
                    }
                    
                    if(j % 2 == 0){
                        float2 p1 = {uMin + (i * du), vMin + (j* dv)};
                        float2 p2 = {uMin + (i * du), vMin + ((j+2)* dv)};
                        float2 p3 = {uMin + ((i+2) * du), vMin + (j* dv)};
                        
                        float2 centroid = {(p1.x + p2.x + p3.x)/3, (p1.y + p2.y + p3.y)/3};
                        
                        float3 normal = unitNormal(centroid.x, centroid.y, st, animate);
                    
                        out[((i*width + j)*3)] = normal.x;
                        out[((i*width + j)*3) + 1] = normal.y;
                        out[((i*width + j)*3) + 2] = normal.z;
 
                    }
                    else{
                         float2 p1 = {uMin + (i * du), vMin + ((j+1)* dv)};
                        
                        float2 p2 = {uMin + ((i+2) * du), vMin + ((j-1)* dv)};
                        float2 p3 = {uMin + ((i+2) * du), vMin + ((j+1)* dv)};
                         
                        float2 centroid = {(p1.x + p2.x + p3.x)/3, (p1.y + p2.y + p3.y)/3};
                        
                        float3 normal = unitNormal(centroid.x, centroid.y, st, animate);
                    
                        out[((i*width + j)*3)] = normal.x;
                        out[((i*width + j)*3) + 1] = normal.y;
                        out[((i*width + j)*3) + 2] = normal.z;

                    }
                }

            }
            __global__ void compute_tris_colors(float* out, float* verts){

                int width = (int) ((vMax - vMin) / dv);
                int i = threadIdx.x;

                for(int j = 0; j < width; ++j){
                     if(i == 0 || j == 0 || i >= width-3 || j >= width - 3){
                        continue;
                    }
                    
                    if(j % 2 == 0){
                        out[(i*width + j)*4*3] = verts[(i*width + j)*4];
                        out[(i*width + j)*4*3 + 1] = verts[(i*width +j)*4 + 1];
                        out[(i*width + j)*4*3 + 2] = verts[(i * width + j)*4 + 2];
                        out[(i*width + j)*4*3 + 3] = verts[(i * width + j)*4 + 3];
                        
                        out[((i*width + j)*3 + 1)*4] = verts[(i*width + j+2)*4];
                        out[((i*width + j)*3 + 1)*4 + 1] = verts[(i*width + j+2)*4 + 1];
                        out[((i*width + j)*3 + 1)*4 + 2] = verts[(i*width + j+2)*4 + 2];
                        out[((i*width + j)*3 + 1)*4 + 3] = verts[(i*width + j+2)*4 + 3];
 
                        out[((i*width + j)*3 +2)*4] = verts[((i+2)*width + j)*4];
                        out[((i*width + j)*3 +2)*4 + 1] = verts[((i+2)*width +j)*4 + 1];
                        out[((i*width + j)*3 +2)*4 + 2] = verts[((i+2) * width + j)*4 + 2];
                        out[((i*width + j)*3 +2)*4 + 3] = verts[((i+2) * width + j)*4 + 3];
                    }
                    else{
                        out[(i*width + j)*4*3] = verts[(i*width + j+1)*4];
                        out[(i*width + j)*4*3 + 1] = verts[(i*width +j+1)*4 + 1];
                        out[(i*width + j)*4*3 + 2] = verts[(i * width + j+1)*4 + 2];
                        out[(i*width + j)*4*3 + 2] = verts[(i * width + j+1)*4 + 3];
                        
                        out[((i*width + j)*3 + 1)*4] = verts[((i+2)*width + j-1)*4];
                        out[((i*width + j)*3 + 1)*4 + 1] = verts[((i+2)*width + j-1)*4 + 1];
                        out[((i*width + j)*3 + 1)*4 + 2] = verts[((i+2)*width + j-1)*4 + 2];
                        out[((i*width + j)*3 + 1)*4 + 3] = verts[((i+2)*width + j-1)*4 + 3];
 
                        out[((i*width + j)*3 +2)*4] = verts[((i+2)*width + j+1)*4];
                        out[((i*width + j)*3 +2)*4 + 1] = verts[((i+2)*width +j+1)*4 + 1];
                        out[((i*width + j)*3 +2)*4 + 2] = verts[((i+2) * width + j+1)*4 + 2];
                        out[((i*width + j)*3 +2)*4 + 3] = verts[((i+2) * width + j+1)*4 + 3];
                    }
               
                
                
                
                }


            }
            __global__ void compute_colors(float* out, float* time, bool square, int curvature_type, bool animate) {
                
                int width = (int) ((vMax - vMin) / dv);

                float st = time[0];
                
                int i = threadIdx.x;
                
                for(int j = 0; j < width; ++j){
                    
                    float up = uMin + (i * du);
                    float vp = vMin + (j * dv);
                    float u = up;
                    float v = vp;
                    if(!square){
                        u = up*cosf(vp);
                        v = up*sinf(vp);
                    }
                    float r = 0;
                    float g = 1;
                    float b = 0;
                    if(curvature_type == 1){
                        uVel = uVelocity(u, v, st, animate);
                        vVel = vVelocity(u, v, st, animate);
                        float gc = gaussianCurvature(u, v, st, animate);
                        r = gc;
                        g = 1-gc;
                    }else if(curvature_type == 2){
                        uVel = uVelocity(u, v, st, animate);
                        vVel = vVelocity(u, v, st, animate);
                     
                        float nc = meanCurvature(u, v, st, animate);
                        r = nc;
                        g = 1-nc;
                    }
                    
                    out[(i*width + j)*3] = r;
                    
                    out[(i*width + j)*3 + 1] = g;
                    
                    out[(i*width + j)*3 + 2] = b;
                }

            }

            """)

        self.f_compute = self.gpu_mod.get_function('compute_verts')
        self.f_compute_colors = self.gpu_mod.get_function('compute_colors')
        self.f_compute_tris = self.gpu_mod.get_function('compute_tris')
        self.f_compute_tris_colors = self.gpu_mod.get_function('compute_tris_colors')
        self.f_compute_normals = self.gpu_mod.get_function('compute_tris_normals')
        
        height = int((uMax - uMin) // du)
        width = int((vMax - vMin) // dv)
        self.height = height

        self.verts = cuda.mem_alloc(int(height*width*32*3))
        self.colors = cuda.mem_alloc(int(height*width*32*3))
        self.tris = cuda.mem_alloc(int(height*width*32*3*3))
        self.tris_colors = cuda.mem_alloc(int(height*width*32*6*4))
        self.normals = cuda.mem_alloc(int(height*width*32*3))

        self.time = cuda.mem_alloc(32*3)
        
        cuda.memcpy_htod(self.time, np.array([0, 0, 0]).astype(np.float32))

        self.f_compute(self.verts, self.time, np.int8(self.is_square), np.int8(self.animate), block = (height, 1, 1))
        self.f_compute_tris(self.tris, self.verts, block = (height, 1, 1))
        self.f_compute_colors(self.colors, self.time, np.int8(self.is_square), np.int32(self.curvature_type), np.int8(self.animate), block = (height, 1, 1))
        self.f_compute_tris_colors(self.tris_colors, self.colors, block = (height, 1, 1))
        self.f_compute_normals(self.normals, self.time, np.int8(self.animate), block = (height, 1, 1))
        
        context.synchronize()
        self.verts_cpu = np.zeros(height*width*3).astype(np.float32)
        
        self.tris_cpu = np.zeros(height*width*3*3).astype(np.float32)
        
        self.colors_cpu = np.zeros(height*width*3).astype(np.float32)
        
        self.tris_colors_cpu = np.zeros(height*width*6*4).astype(np.float32)
        
        self.normals_cpu = np.zeros(height*width*3).astype(np.float32)

        cuda.memcpy_dtoh(self.verts_cpu, self.verts)
        cuda.memcpy_dtoh(self.colors_cpu, self.colors)
        cuda.memcpy_dtoh(self.tris_cpu, self.tris)
        cuda.memcpy_dtoh(self.tris_colors_cpu, self.tris_colors)
        cuda.memcpy_dtoh(self.normals_cpu, self.normals)
        context.synchronize()

    def get_normals(self):
        return self.normals_cpu
    def get_verts(self):
        return self.verts_cpu
    
    def get_colors(self):
        return self.colors_cpu
    def get_tris(self):
        return self.tris_cpu

    def compute(self):
        if not self.updates:
            return
        if not self.is_paused:
            self.ticks += 1*self.time_direction
            self._time = self.ticks*0.01
            if self._time > self.time_bounds[1] or self._time < self.time_bounds[0]:
                self.time_direction = -self.time_direction
        cuda.memcpy_htod(self.time, np.array([self._time, 0, 0]).astype(np.float32))
        context.synchronize()
        self.f_compute(self.verts, self.time, np.int8(self.is_square), np.int8(self.animate), block = (self.height, 1, 1))

        self.f_compute_tris(self.tris, self.verts, block = (self.height, 1, 1))
        self.f_compute_colors(self.colors, self.time, np.int8(self.is_square), np.int32(self.curvature_type), np.int8(self.animate), block = (self.height, 1, 1))
        self.f_compute_tris_colors(self.tris_colors, self.colors, block = (self.height, 1, 1)) 
        self.f_compute_normals(self.normals, self.time, np.int8(self.animate), block = (self.height, 1, 1))
        
        cuda.memcpy_dtoh(self.colors_cpu, self.colors)
        cuda.memcpy_dtoh(self.tris_cpu, self.tris)
        cuda.memcpy_dtoh(self.verts_cpu, self.verts)
        cuda.memcpy_dtoh(self.tris_colors_cpu, self.tris_colors)
        cuda.memcpy_dtoh(self.normals_cpu, self.normals)
        context.synchronize()

class TestPatch(GPUPatch):
    def __init__(self, updates):
        super(TestPatch, self).__init__(bounds = (0, 6.28, 0, 6.28), updates=updates)

    def patch_function(self):
        return """ 
        phii(float u, float v, float t){
            return {u, v, (1-t)*cos(u)*sin(v)};
        }
        """

class FourierPatch(GPUPatch):
    def __init__(self, updates, time_bounds=(0, 1)):
        self.patch_func = rand_interpolate_fourier_series(3)
        super(FourierPatch, self).__init__(bounds = (-3.14, 3.14, -3.14, 3.14), updates=updates, time_bounds=time_bounds)
    
    def phi_x(self):
        return 'u'
    def phi_y(self):
        return 'v'
    def phi_z(self):
        return self.patch_func
    def u_bounds(self):
        return '(-3.14, 3.14)'
    def v_bounds(self):
        return '(-3.14, 3.14)'

    def patch_function(self):
        return """
        phii(float u, float v, float t){{
            return {{u, v, {}}};
        }}
        """.format(self.patch_func)

class StringPatch(GPUPatch):
    def __init__(self, patch_string, updates = False, bounds_string = "(0, 1, 0, 1)", time_bounds = (0, 1)):
        self.patch_string = patch_string
        bounds_tuple = tuple(map(float, bounds_string.strip('(').strip(')').split(',')))
        super(StringPatch, self).__init__(bounds = bounds_tuple, updates=updates, time_bounds=time_bounds)
        
    def patch_function(self):
        return """
            phii(float u, float v, float t){{
                return {{ {} }};
            }}
        """.format(self.patch_string)
