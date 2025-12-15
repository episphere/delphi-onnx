# JavaScript SDK for Delphi-2M (+ ONNX Export scripts)

### Example Usage
> Trajectory Generation:
```js
let eventsList = [
    {
        "event": "Male",
        "age": 0
    },
    {
        "event": "B01 Varicella [chickenpox]",
        "age": 2
    },
    {
        "event": "L20 Atopic dermatitis",
        "age": 3
    },
    {
        "event": "No event",
        "age": 5
    },
    {
        "event": "No event",
        "age": 10
    },
    {
        "event": "No event",
        "age": 15
    },
    {
        "event": "No event",
        "age": 20
    },
    {
        "event": "G43 Migraine",
        "age": 20
    },
    {
        "event": "E73 Lactose intolerance",
        "age": 21
    },
    {
        "event": "B27 Infectious mononucleosis",
        "age": 22
    },
    {
        "event": "No event",
        "age": 25
    },
    {
        "event": "J11 Influenza, virus not identified",
        "age": 28
    },
    {
        "event": "No event",
        "age": 30
    },
    {
        "event": "No event",
        "age": 35
    },
    {
        "event": "No event",
        "age": 40
    },
    {
        "event": "Smoking low",
        "age": 41
    },
    {
        "event": "BMI mid",
        "age": 41
    },
    {
        "event": "Alcohol low",
        "age": 41
    },
    {
        "event": "No event",
        "age": 42
    }
]

const { generateTrajectory } = await import("https://episphere.github.io/delphi-onnx/delphiSDK.js")
await generateTrajectory({eventsList, seed: 42})
```

> Embeddings:
```js
let eventsList = [
    {
        "event": "Male",
        "age": 0
    },
    {
        "event": "B01 Varicella [chickenpox]",
        "age": 2
    },
    {
        "event": "L20 Atopic dermatitis",
        "age": 3
    },
    {
        "event": "No event",
        "age": 5
    },
    {
        "event": "No event",
        "age": 10
    },
    {
        "event": "No event",
        "age": 15
    },
    {
        "event": "No event",
        "age": 20
    },
    {
        "event": "G43 Migraine",
        "age": 20
    },
    {
        "event": "E73 Lactose intolerance",
        "age": 21
    },
    {
        "event": "B27 Infectious mononucleosis",
        "age": 22
    },
    {
        "event": "No event",
        "age": 25
    },
    {
        "event": "J11 Influenza, virus not identified",
        "age": 28
    },
    {
        "event": "No event",
        "age": 30
    },
    {
        "event": "No event",
        "age": 35
    },
    {
        "event": "No event",
        "age": 40
    },
    {
        "event": "Smoking low",
        "age": 41
    },
    {
        "event": "BMI mid",
        "age": 41
    },
    {
        "event": "Alcohol low",
        "age": 41
    },
    {
        "event": "No event",
        "age": 42
    }
]

const { getEmbeddings } = await import("https://episphere.github.io/delphi-onnx/delphiSDK.js")
await getEmbeddings({eventsList, pooling: 'mean'}) // pooling could be mean, max, or last. If not specified, embeddings for all events will be returned.
```

To run the export script:
> For the predictive model:
`python export_onnx.py --checkpoint ckpt.pt --output model.onnx`

> For the embeddings only model:
`python export_onnx.py --checkpoint ckpt.pt --output embeddingsModel.onnx --embeddings-only`