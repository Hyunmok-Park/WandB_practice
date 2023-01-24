from torchvision import transforms

hyperparameter_defaults  = {
        'epochs': 2,
        'batch_size': 128,
        'dropout': 0.3,
        'learning_rate': 1e-3,
        'optimizer': 'adam',
    }

sweep_config = {
    'name' : 'bayes-test',
    'method': 'random',
    'metric' : {
        'name': 'valid_loss',
        'goal': 'minimize'
        },
    'parameters' : {
        'optimizer': {
            'values': ['adam', 'sgd']
            },
        'epochs': {
            'values': [3, 4]
            },
        'learning_rate': {
            'values': [0.1, 0.01]
            },
        'batch_size': {
            'values': [32, 64]
            }
        }
    }

train_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])