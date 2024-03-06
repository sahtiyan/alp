#include <QtWidgets>
#include <nvml.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
do { \
    cudaError_t result = call; \
    if (result != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s)!\n", __FILE__, __LINE__, result, cudaGetErrorString(result)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

class GPUInfoWidget : public QWidget {
    Q_OBJECT

public:
    GPUInfoWidget(QWidget *parent = nullptr) : QWidget(parent) {
        QVBoxLayout *layout = new QVBoxLayout(this);
        
        // Başlık
        QLabel *titleLabel = new QLabel("<h2 style=\"color:white;\">Alp - GPU Bilgi Uygulaması</h2>");
        layout->addWidget(titleLabel);

        // Genel GPU Bilgileri
        gpuInfoLabel = new QLabel();
        updateGPUInfo(gpuInfoLabel);
        layout->addWidget(gpuInfoLabel);

        // CUDA Kernels Bilgileri
        QPushButton *kernelInfoButton = new QPushButton("CUDA Kernels Bilgisi");
        layout->addWidget(kernelInfoButton);
        connect(kernelInfoButton, &QPushButton::clicked, this, &GPUInfoWidget::showKernelInfo);

        // Özellikler
        setStyleSheet("background-color: #2b2b2b; color: white;");
    }

    void updateGPUInfo(QLabel *label) {
        nvmlInit();
        nvmlDevice_t device;
        nvmlDeviceGetHandleByIndex(0, &device); // Sadece bir GPU varsa
        char name[NVML_DEVICE_NAME_BUFFER_SIZE];
        nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
        nvmlMemory_t mem;
        nvmlDeviceGetMemoryInfo(device, &mem);
        nvmlUtilization_t util;
        nvmlDeviceGetUtilizationRates(device, &util);
        nvmlShutdown();

        QString info = QString("<b>GPU Adı:</b> %1<br>").arg(name);
        info += QString("<b>Bellek Kullanımı:</b> %1 MB / %2 MB<br>").arg(mem.used / 1024 / 1024).arg(mem.total / 1024 / 1024);
        info += QString("<b>% GPU Kullanımı:</b> %1<br>").arg(util.gpu);
        info += QString("<b>% Bellek Kullanımı:</b> %1<br>").arg(util.memory);

        label->setText(info);
    }

signals:
    void showKernelInfo();

private slots:
    void showKernelInfo() {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        int *d_array;
        CUDA_CHECK(cudaMalloc(&d_array, 1000 * sizeof(int)));

        CUDA_CHECK(cudaEventRecord(start));
        kernel<<<1, 1>>>(d_array);
        CUDA_CHECK(cudaEventRecord(stop));

        CUDA_CHECK(cudaEventSynchronize(stop));

        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

        QLabel *kernelInfoLabel = new QLabel(QString("<h2 style=\"color:white;\">CUDA Kernel Zamanlaması</h2>"));
        kernelInfoLabel->append(QString("Kernel Çalışma Süresi: %1 ms").arg(milliseconds));
        QVBoxLayout *layout = qobject_cast<QVBoxLayout*>(this->layout());
        layout->addWidget(kernelInfoLabel);
        
        CUDA_CHECK(cudaFree(d_array));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

private:
    QLabel *gpuInfoLabel;
};

int main(int argc, char **argv) {
    QApplication app(argc, argv);

    GPUInfoWidget widget;
    widget.show();

    return app.exec();
}

#include "main.moc"
