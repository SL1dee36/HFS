#include <GLFW/glfw3.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <algorithm>
#include <thread>
#include <mutex>
#include <immintrin.h>
#include <chrono>
#include <memory>

// Настройки
const int WIDTH = 770;
const int HEIGHT = 500;
const int FPS = 5000;
const int PARTICLE_RADIUS = 5;
float GRAVITY = 0.0981f;
float REST_DENSITY = 10.0f;
float GAS_CONSTANT = 2.0f;
float VISCOSITY = 0.9f;
int PARTICLE_CREATION_RATE = 100;
float COHESION_STRENGTH = 0.2f;
float DAMPING = 0.99f;
float DRAG_COEFFICIENT = 0.1f;
const int GRAB_RADIUS = 50;
float SPRING_CONSTANT = 0.2f;
const int MAX_PARTICLES = 9000; // Ограничение на количество частиц
float ROTATION_SPEED = 20.0f; // Скорость вращения частиц
int IMPULSE_STRENGTH = 50; // Сила импульса (теперь переменная)

#define DEG2RAD 0.017453292519943295f // PI / 180


// Препятствие
const int OBSTACLE_X = 200;
const int OBSTACLE_Y = 200;
const int OBSTACLE_RADIUS = 0;

// Стены
const int WALL_THICKNESS = 0;
struct Color { unsigned char r, g, b, a; }; // Замена SDL_Color
const Color WALL_COLOR = { 255, 255, 255, 255 };
const int PAD_X = 240;
const int PAD_Y = 20;

// Градиент цветов
const int GRADIENT_STEPS = 4096;
Color color_gradient[GRADIENT_STEPS];
const float max_speed = 16.0f;

// Количество потоков
const int NUM_THREADS = std::thread::hardware_concurrency();

// Мьютекс для синхронизации доступа к particles
std::mutex particles_mutex;

// Флаг для отображения стрелок
bool show_arrows = false;

// Переменная для хранения состояния дебаг-меню
bool debug_mode = false;


// Замер времени
using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;

class Quadtree {
public:
    // Границы квадродерева
    float x, y, width, height;
    // Максимальное количество частиц в узле
    int capacity;
    // Вектор частиц в данном узле
    std::vector<Particle*> particles;
    // Дочерние узлы
    std::unique_ptr<Quadtree> nw, ne, sw, se;
    bool divided;


    Quadtree(float x, float y, float width, float height, int capacity) :
        x(x), y(y), width(width), height(height), capacity(capacity), divided(false) {}

    void insert(Particle* particle) {
        if (!contains(particle)) return;

        if (particles.size() < capacity) {
            particles.push_back(particle);
        }
        else {
            if (!divided) subdivide();

            nw->insert(particle);
            ne->insert(particle);
            sw->insert(particle);
            se->insert(particle);
        }
    }

    void query(float x, float y, float radius, std::vector<Particle*>& found) {
        if (!intersects(x, y, radius)) return;

        for (Particle* p : particles) {
            float dx = p->x - x;
            float dy = p->y - y;
            if (dx * dx + dy * dy < radius * radius) {
                found.push_back(p);
            }
        }

        if (divided) {
            nw->query(x, y, radius, found);
            ne->query(x, y, radius, found);
            sw->query(x, y, radius, found);
            se->query(x, y, radius, found);
        }
    }


private:

    bool contains(Particle* particle) {
        return particle->x >= x && particle->x < x + width &&
            particle->y >= y && particle->y < y + height;
    }


    bool intersects(float x, float y, float radius) {
        float distX = std::abs(x - (this->x + this->width / 2));
        float distY = std::abs(y - (this->y + this->height / 2));

        if (distX > this->width / 2 + radius) return false;
        if (distY > this->height / 2 + radius) return false;

        if (distX <= this->width / 2) return true;
        if (distY <= this->height / 2) return true;

        float cornerDistSq = (distX - this->width / 2) * (distX - this->width / 2) +
            (distY - this->height / 2) * (distY - this->height / 2);

        return cornerDistSq <= radius * radius;

    }


    void subdivide() {
        float halfWidth = width / 2;
        float halfHeight = height / 2;

        nw = std::make_unique<Quadtree>(x, y, halfWidth, halfHeight, capacity);
        ne = std::make_unique<Quadtree>(x + halfWidth, y, halfWidth, halfHeight, capacity);
        sw = std::make_unique<Quadtree>(x, y + halfHeight, halfWidth, halfHeight, capacity);
        se = std::make_unique<Quadtree>(x + halfWidth, y + halfHeight, halfWidth, halfHeight, capacity);

        divided = true;
    }
};




// Функция для создания градиента цветов
void create_gradient(Color color1, Color color2, int steps, Color* gradient) {
    for (int i = 0; i < steps; ++i) {
        float t = i / (steps - 1.0f);
        gradient[i].r = static_cast<unsigned char>(color1.r * (1 - t) + color2.r * t);
        gradient[i].g = static_cast<unsigned char>(color1.g * (1 - t) + color2.g * t);
        gradient[i].b = static_cast<unsigned char>(color1.b * (1 - t) + color2.b * t);
        gradient[i].a = 255;
    }
}


class Particle {
public:
    float x;
    float y;
    float vx;
    float vy;
    float density;
    float pressure;
    float angle; // Угол поворота
    float angular_velocity; // Угловая скорость
    std::vector<Particle*> near_particles;
    bool grabbed;

    Particle(float x, float y) : x(x), y(y), vx(0), vy(0), density(0), pressure(0), angle(0), angular_velocity(0), grabbed(false) {}

    void update() {
        vy += GRAVITY;
        x += vx;
        y += vy;

        // Обновление вращения
        angle += angular_velocity;

        // Столкновения с границами (оптимизировано)
        if (x < PARTICLE_RADIUS + PAD_X) {
            x = PARTICLE_RADIUS + PAD_X;
            vx *= -0.7f;
        }
        else if (x > WIDTH - PARTICLE_RADIUS - PAD_X) {
            x = WIDTH - PARTICLE_RADIUS - PAD_X;
            vx *= -0.7f;
        }

        if (y > HEIGHT - PARTICLE_RADIUS - PAD_Y) {
            y = HEIGHT - PARTICLE_RADIUS - PAD_Y;
            vy *= -0.7f;
        }

        // Добавлено условие для верхней границы
        if (y < PARTICLE_RADIUS + PAD_Y) {
            y = PARTICLE_RADIUS + PAD_Y;
            vy *= -0.7f;
        }

        // Столкновение с препятствием (оптимизировано)
        float dx = x - OBSTACLE_X;
        float dy = y - OBSTACLE_Y;
        float distance_to_obstacle_sq = dx * dx + dy * dy;
        float radius_sum_sq = (PARTICLE_RADIUS + OBSTACLE_RADIUS) * (PARTICLE_RADIUS + OBSTACLE_RADIUS);

        if (distance_to_obstacle_sq < radius_sum_sq) {
            float distance_to_obstacle = std::sqrt(distance_to_obstacle_sq);
            float overlap = (PARTICLE_RADIUS + OBSTACLE_RADIUS) - distance_to_obstacle;
            x += overlap * dx / distance_to_obstacle;
            y += overlap * dy / distance_to_obstacle;
            vx *= -0.7f;
            vy *= -0.7f;
        }

        // Демпфирование скорости
        vx *= DAMPING;
        vy *= DAMPING;
    }


    void draw() {
        // Использование градиента для визуализации скорости
        float speed = std::sqrt(vx * vx + vy * vy);
        int colorIndex = static_cast<int>(speed / max_speed * (GRADIENT_STEPS - 1));
        colorIndex = std::min(colorIndex, GRADIENT_STEPS - 1);

        // Рисование круга с помощью OpenGL
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glColor4ub(color_gradient[colorIndex].r, color_gradient[colorIndex].g, color_gradient[colorIndex].b, 128);
        glBegin(GL_POLYGON);

        // Выбор количества сегментов в зависимости от debug_mode
        int num_segments = debug_mode ? 0 : 32;

        for (int i = 0; i < num_segments; i++) {
            float degInRad = i * 2 * M_PI / num_segments;
            glVertex2f(x + cos(degInRad) * PARTICLE_RADIUS, y + sin(degInRad) * PARTICLE_RADIUS);
        }
        glEnd();

        glDisable(GL_BLEND);
    }

    void reset_velocity() {
        vx = 0;
        vy = 0;
        angular_velocity = 0; // Сброс угловой скорости
    }


    void apply_spring_force(Particle* other) {
        float dx = x - other->x;
        float dy = y - other->y;
        float distance_sq = dx * dx + dy * dy;
        float radius_sum_sq = (2 * PARTICLE_RADIUS) * (2 * PARTICLE_RADIUS);

        if (distance_sq > radius_sum_sq) {
            float distance = std::sqrt(distance_sq);
            float force = SPRING_CONSTANT * (distance - 2 * PARTICLE_RADIUS);
            float fx = force * dx / distance;
            float fy = force * dy / distance;
            vx -= fx / density;
            vy -= fy / density;
            other->vx += fx / other->density;
            other->vy += fy / other->density;

            // Добавление вращения
            float torque = (fx * dy - fy * dx) * ROTATION_SPEED;
            angular_velocity += torque / density;
            other->angular_velocity -= torque / other->density;
        }
    }

    // Применение импульса к частице
    void apply_impulse(float impulse_x, float impulse_y) {
        vx += impulse_x / density;
        vy += impulse_y / density;
    }
};


void find_neighbors(std::vector<Particle>& particles, Quadtree& qtree) {
    qtree.nw = nullptr; // Очистка дочерних узлов перед построением нового дерева
    qtree.ne = nullptr;
    qtree.sw = nullptr;
    qtree.se = nullptr;
    qtree.particles.clear(); // Очистка вектора частиц в корневом узле
    qtree.divided = false;


    for (Particle& p : particles) {
        qtree.insert(&p);
    }

    for (Particle& p : particles) {
        p.near_particles.clear();
        qtree.query(p.x, p.y, 2 * PARTICLE_RADIUS, p.near_particles);
    }
}


// Оптимизированная функция для вычисления плотности и давления
void calculate_density_pressure(std::vector<Particle>& particles, int start, int end) {
    for (size_t i = start; i < end; ++i) {
        particles[i].density = 0;
        for (Particle* other : particles[i].near_particles) {
            float dx = particles[i].x - other->x;
            float dy = particles[i].y - other->y;
            float distance_sq = dx * dx + dy * dy;
            float radius_sum_sq = (2 * PARTICLE_RADIUS) * (2 * PARTICLE_RADIUS);

            if (distance_sq < radius_sum_sq) {
                particles[i].density += 1;
            }
        }
        particles[i].density = std::max(particles[i].density, 0.1f);
        particles[i].pressure = GAS_CONSTANT * (particles[i].density - REST_DENSITY);
    }
}

// Оптимизированная функция для вычисления сил
void calculate_forces(std::vector<Particle>& particles, int start, int end) {
    for (size_t i = start; i < end; ++i) {
        float dx = 0, dy = 0;
        for (Particle* other : particles[i].near_particles) {
            float dx_diff = particles[i].x - other->x;
            float dy_diff = particles[i].y - other->y;
            float distance_sq = dx_diff * dx_diff + dy_diff * dy_diff;
            float radius_sum_sq = (2 * PARTICLE_RADIUS) * (2 * PARTICLE_RADIUS);

            if (distance_sq < radius_sum_sq && distance_sq > 0) {
                float distance = std::sqrt(distance_sq);

                // Сила давления
                float pressure_force = (particles[i].pressure + other->pressure) / 2 * (1 - distance / (2 * PARTICLE_RADIUS));
                dx += pressure_force * dx_diff / distance;
                dy += pressure_force * dy_diff / distance;

                // Сила вязкости
                float vx_diff = other->vx - particles[i].vx;
                float vy_diff = other->vy - particles[i].vy;
                float viscosity_force = VISCOSITY * (vx_diff * dx_diff + vy_diff * dy_diff) / distance_sq;
                dx += viscosity_force * dx_diff;
                dy += viscosity_force * dy_diff;

                // Сила сцепления
                float cohesion_force = COHESION_STRENGTH / distance;
                dx -= cohesion_force * dx_diff;
                dy -= cohesion_force * dy_diff;


                particles[i].apply_spring_force(other);
            }
        }

        // Применение сил
        particles[i].vx += dx / particles[i].density;
        particles[i].vy += dy / particles[i].density;
    }
}


void update_physics_thread(std::vector<Particle>& particles, int start, int end) {
    calculate_density_pressure(particles, start, end);
    calculate_forces(particles, start, end);
}




struct Slider {
    std::string label;
    float* value;
    float min_value;
    float max_value;
    int x, y, width, height;
};

void draw_slider(Slider slider) {
    // Отрисовка фона ползунка
    glColor4ub(200, 200, 200, 255);
    glBegin(GL_QUADS);
    glVertex2f(slider.x, slider.y);
    glVertex2f(slider.x + slider.width, slider.y);
    glVertex2f(slider.x + slider.width, slider.y + slider.height);
    glVertex2f(slider.x, slider.y + slider.height);
    glEnd();

    // Вычисление позиции ползунка
    float slider_pos = slider.x + slider.width * (*slider.value - slider.min_value) / (slider.max_value - slider.min_value);

    // Отрисовка ползунка
    glColor4ub(100, 100, 100, 255);
    glBegin(GL_QUADS);
    glVertex2f(slider_pos - 5, slider.y);
    glVertex2f(slider_pos + 5, slider.y);
    glVertex2f(slider_pos + 5, slider.y + slider.height);
    glVertex2f(slider_pos - 5, slider.y + slider.height);
    glEnd();
}




bool handle_slider_event(Slider slider, double mouseX, double mouseY) {
    if (mouseX >= slider.x && mouseX <= slider.x + slider.width &&
        mouseY >= slider.y && mouseY <= slider.y + slider.height) {
        *slider.value = slider.min_value + (mouseX - slider.x) * (slider.max_value - slider.min_value) / slider.width;
        return true;
    }
    return false;
}





int main(int argc, char* argv[]) {
    // Инициализация GLFW
    if (!glfwInit()) {
        std::cerr << "glfwInit failed" << std::endl;
        return 1;
    }

    glfwWindowHint(GLFW_DEPTH_BITS, 32);
    glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);

    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "2D Fluid Simulation", nullptr, nullptr);
    if (!window) {
        std::cerr << "glfwCreateWindow failed" << std::endl;
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, WIDTH, HEIGHT, 0, 1, -1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    Color color1 = { 0, 0, 255, 255 }; // Синий
    Color color2 = { 255, 255, 255, 255 }; // Желтый
    create_gradient(color1, color2, GRADIENT_STEPS, color_gradient);


    std::vector<Particle> particles;

    bool mouse_pressed_left = false;
    bool mouse_pressed_right = false;
    bool mouse_pressed_central = false;

    std::vector<Particle*> grabbed_particles;


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(-PARTICLE_RADIUS * 2, PARTICLE_RADIUS * 2);


    std::vector<Slider> sliders = {
        {"Gravity", &GRAVITY, 0.0f, 1.0f, 10, 10, 100, 20},
        {"Rest Density", &REST_DENSITY, 1.0f, 20.0f, 10, 40, 100, 20},
        {"Gas Constant", &GAS_CONSTANT, 0.1f, 5.0f, 10, 70, 100, 20},
        {"Viscosity", &VISCOSITY, 0.1f, 2.0f, 10, 100, 100, 20},
        {"Cohesion", &COHESION_STRENGTH, 0.0f, 1.0f, 10, 130, 100, 20},
        {"Spring Constant", &SPRING_CONSTANT, 0.0f, 1.0f, 10, 160, 100, 20},
    };

    Quadtree qtree(PAD_X, PAD_Y, WIDTH - 2 * PAD_X, HEIGHT - 2 * PAD_Y, 4);



    Duration find_neighbors_time(0);
    Duration density_pressure_time(0);
    Duration forces_time(0);
    Duration update_time(0);
    Duration draw_time(0);
    Duration frame_time(0);


    double previousTime = glfwGetTime();
    int frameCount = 0;
    int fps_int = 0; // Инициализация fps_int


    while (!glfwWindowShouldClose(window)) {

        double currentTime = glfwGetTime();
        frameCount++;

        if (currentTime - previousTime >= 1.0) {
            fps_int = static_cast<int>(frameCount / (currentTime - previousTime));
            frameCount = 0;
            previousTime = currentTime;
        }


        glfwPollEvents();


        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);


        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
            mouse_pressed_left = true;
            for (Slider& slider : sliders) {
                handle_slider_event(slider, xpos, ypos);
            }

        }
        else {
            mouse_pressed_left = false;
        }



        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {

            if (!mouse_pressed_right) { // проверка на первое нажатие 
                mouse_pressed_right = true;
                for (Particle& particle : particles) {
                    float dx = particle.x - xpos;
                    float dy = particle.y - ypos;
                    float distance_sq = dx * dx + dy * dy;
                    float grab_radius_sq = GRAB_RADIUS * GRAB_RADIUS;

                    if (distance_sq <= grab_radius_sq) {
                        particle.grabbed = true;
                        grabbed_particles.push_back(&particle);
                    }
                }
            }
        }
        else {
            mouse_pressed_right = false;
            for (Particle* particle : grabbed_particles) {
                particle->grabbed = false;
            }
            grabbed_particles.clear();
        }



        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS) {
            if (!mouse_pressed_central) { // проверка на первое нажатие
                mouse_pressed_central = true;

                for (Particle& particle : particles) {
                    float dx = particle.x - xpos;
                    float dy = particle.y - ypos;
                    float distance_sq = dx * dx + dy * dy;
                    float impulse_radius_sq = GRAB_RADIUS * GRAB_RADIUS;

                    if (distance_sq <= impulse_radius_sq) {
                        float distance = std::sqrt(distance_sq);
                        // Расчет импульса в зависимости от расстояния
                        float impulse_strength = IMPULSE_STRENGTH * (1 - distance / GRAB_RADIUS);
                        float impulse_x = impulse_strength * dx / distance;
                        float impulse_y = impulse_strength * dy / distance;
                        particle.apply_impulse(impulse_x, impulse_y);
                    }
                }
            }

        }
        else {
            mouse_pressed_central = false;
        }






        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
            for (Particle& particle : particles) {
                particle.reset_velocity();
            }
        }


        if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
            particles.clear();
        }


        if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
            IMPULSE_STRENGTH += 5.0f;
        }



        if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
            IMPULSE_STRENGTH = std::max(0.0f, IMPULSE_STRENGTH - 5.0f);
        }



        if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) {
            debug_mode = !debug_mode;
            show_arrows = !show_arrows;
        }


        if (glfwGetKey(window, GLFW_KEY_0) == GLFW_PRESS) {
            debug_mode = false; // Выключаем дебаг-режим
            show_arrows = false;
        }



        if (mouse_pressed_left && particles.size() < MAX_PARTICLES) {
            for (int i = 0; i < PARTICLE_CREATION_RATE; ++i) {
                float x = xpos + distrib(gen);
                float y = ypos + distrib(gen);
                particles_mutex.lock();
                particles.emplace_back(x, y);
                particles_mutex.unlock();
            }
        }



        if (mouse_pressed_right) {
            for (Particle* particle : grabbed_particles) {
                float dx = xpos - particle->x;
                float dy = ypos - particle->y;
                particle->vx = dx * DRAG_COEFFICIENT;
                particle->vy = dy * DRAG_COEFFICIENT;
            }
        }





        auto start_time = Clock::now();

        particles_mutex.lock();
        find_neighbors(particles, qtree);
        particles_mutex.unlock();


        auto end_physics_time = Clock::now();


        std::vector<std::thread> threads;
        int particles_per_thread = particles.size() / NUM_THREADS;
        for (int i = 0; i < NUM_THREADS; ++i) {
            int start = i * particles_per_thread;
            int end = (i == NUM_THREADS - 1) ? particles.size() : (i + 1) * particles_per_thread;
            threads.push_back(std::thread(update_physics_thread, std::ref(particles), start, end));
        }

        for (auto& thread : threads) {
            thread.join();
        }



        particles_mutex.lock();
        for (Particle& particle : particles) {
            particle.update();
        }
        particles_mutex.unlock();
        auto end_update_time = Clock::now();


        glClear(GL_COLOR_BUFFER_BIT);


        glColor4ub(WALL_COLOR.r, WALL_COLOR.g, WALL_COLOR.b, WALL_COLOR.a);
        glBegin(GL_LINES);
        glVertex2f(PAD_X, PAD_Y);
        glVertex2f(WIDTH - PAD_X, PAD_Y);
        glVertex2f(PAD_X, HEIGHT - PAD_Y);
        glVertex2f(WIDTH - PAD_X, HEIGHT - PAD_Y);
        glVertex2f(PAD_X, PAD_Y);
        glVertex2f(PAD_X, HEIGHT - PAD_Y);
        glVertex2f(WIDTH - PAD_X, PAD_Y);
        glVertex2f(WIDTH - PAD_X, HEIGHT - PAD_Y);
        glEnd();



        particles_mutex.lock();
        for (Particle& particle : particles) {
            particle.draw();
        }
        particles_mutex.unlock();


        if (show_arrows) {
            particles_mutex.lock();
            for (Particle& particle : particles) {
                float speed = std::sqrt(particle.vx * particle.vx + particle.vy * particle.vy);
                if (speed > 0.1f) {
                    float arrow_length = 10.0f;
                    float arrow_angle = atan2(particle.vy, particle.vx);
                    float arrow_tip_x = particle.x + arrow_length * cos(arrow_angle);
                    float arrow_tip_y = particle.y + arrow_length * sin(arrow_angle);
                    glColor4ub(255, 0, 0, 255);
                    glBegin(GL_LINES);
                    glVertex2f(particle.x, particle.y);
                    glVertex2f(arrow_tip_x, arrow_tip_y);
                    glEnd();
                }
            }
            particles_mutex.unlock();
        }


        for (Slider& slider : sliders) {
            draw_slider(slider);
        }

        auto end_draw_time = Clock::now();


        std::string title = "2D Fluid Simulation v0.975R | Particles: " + std::to_string(particles.size()) + "/" + std::to_string(MAX_PARTICLES) + " | R - clean up | FPS: " + std::to_string(fps_int) + " | Impulse Strength: " + std::to_string(IMPULSE_STRENGTH);
        glfwSetWindowTitle(window, title.c_str());



        auto end_frame_time = Clock::now();


        find_neighbors_time = (end_physics_time - start_time);
        density_pressure_time = (end_update_time - end_physics_time) / NUM_THREADS;
        update_time = end_update_time - end_physics_time;
        draw_time = end_draw_time - end_update_time;
        frame_time = end_frame_time - start_time;




        if (debug_mode) {

            std::cout << "Find Neighbors: " << find_neighbors_time.count() * 1000 << " ms\n";
            std::cout << "Density/Pressure: " << density_pressure_time.count() * 1000 << " ms\n";
            std::cout << "Forces: " << forces_time.count() * 1000 << " ms\n";
            std::cout << "Update: " << update_time.count() * 1000 << " ms\n";
            std::cout << "Draw: " << draw_time.count() * 1000 << " ms\n";
            std::cout << "Frame: " << frame_time.count() * 1000 << " ms\n";
            std::cout << "--------------------\n";
        }

        glfwSwapBuffers(window);

        glfwWaitEventsTimeout(1.0 / FPS);


    }



    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
