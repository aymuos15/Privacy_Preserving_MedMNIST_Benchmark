| Dataset        |   (orig) ACC |   (new) ACC |   (new) ACC [PRIVATE] |   (orig) AUC |   (new) AUC |   (new) AUC [PRIVATE] |
|:---------------|-------------:|------------:|----------------------:|-------------:|------------:|----------------------:|
| breastmnist    |        0.863 |    0.865385 |              0.730769 |        0.901 |    0.882832 |              0.7901   |
| retinamnist    |        0.524 |    0.4975   |              0.545    |        0.717 |    0.707133 |              0.746123 |
| pneumoniamnist |        0.854 |    0.879808 |              0.86859  |        0.944 |    0.959426 |              0.959919 |
| bloodmnist     |        0.958 |    0.940076 |              0.885998 |        0.998 |    0.995822 |              0.986652 |
| organcmnist    |        0.9   |    0.875852 |              0.828019 |        0.992 |    0.987052 |              0.977505 |
| organsmnist    |        0.782 |    0.725615 |              0.679733 |        0.972 |    0.956476 |              0.945694 |
| organamnist    |        0.935 |    0.868489 |              0.824333 |        0.997 |    0.988523 |              0.979146 |
| pathmnist      |        0.911 |    0.820056 |              0.81156  |        0.99  |    0.973512 |              0.959342 |
| octmnist       |        0.743 |    0.737    |              0.667    |        0.943 |    0.930227 |              0.903738 |
| tissuemnist    |        0.676 |    0.55569  |              0.586675 |        0.93  |    0.847001 |              0.84559  |

### Table Legend:
- (orig) Indicates [Original ResNet18 Scores for Mentioned Dataset Size](https://medmnist.com/)
- (new) Indicates Current Run
- (new) [Private] Indicates Current with Privacy

## Experiment Details:

    GPU: NVIDIA GeForce GTX 1080 Ti,
    
    Dataset Size: 28
    batch_size: 1024,
    num_epochs: 120,

    epsilon: 8,
    max_grad_norm: 1.2,

