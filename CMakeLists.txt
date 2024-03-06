cmake_minimum_required(VERSION 3.5)
project(Alp_GPU_Info_App)

# Qt5 kullanımını etkinleştir
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Qt5 bileşenlerini bul
find_package(Qt5 COMPONENTS Widgets REQUIRED)

# CUDA Toolkit'in yolu
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3")

# CUDA kullanımını etkinleştir
find_package(CUDA REQUIRED)

# CUDA dosyalarını ekle
cuda_add_executable(Alp_GPU_Info_App
    src/main.cu
)

# Qt5 kütüphanelerini bağla
target_link_libraries(Alp_GPU_Info_App PRIVATE Qt5::Widgets)
