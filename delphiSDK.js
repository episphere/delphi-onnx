import { InferenceSession, Tensor } from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.mjs"

export const initSession = async () => {
    const session = await InferenceSession.create("https://episphere.github.io/delphi-onnx/delphi.onnx", {
        executionProviders: ["wasm"]
    })
    return session
}

export const getDelphiLogits = async (onnxSession, diseaseTokens, ages, logitsIndex, withDims = false,) => {

    // Only works for a batch size of 1!
    const idxTensor = new onnxRuntime.Tensor(
        "int64",
        new BigInt64Array(diseaseTokens.map((x) => BigInt(x))),
        [1, diseaseTokens.length]
    );

    const ageTensor = new onnxRuntime.Tensor("float32", new Float32Array(ages), [
        1,
        ages.length
    ]);

    const feeds = {
        idx: idxTensor,
        age: ageTensor
    };

    const results = await onnxSession.run(feeds);
    const { data: logits, dims: logitsShape } = results.logits;

    // const batchSize = logitsShape[0]; // Assuming batchSize always 1 for now, index calculation below will need adjustment otherwise
    const numEvents = logitsShape[1];
    const vocabSize = logitsShape[2];

    let logitsForEvent = [];
    // Return last event logits by default
    if (logitsIndex === -1) {
        // Return logits for all events
        logitsForEvent = logits;
    } else {
        let eventLogitsIndex = (numEvents - 1) * vocabSize;
        if (logitsIndex < logitsForEvent.length / vocabSize) {
            eventLogitsIndex = logitsIndex * vocabSize;
        }
        logitsForEvent = logits.slice(
            eventLogitsIndex,
            eventLogitsIndex + vocabSize
        );
    }

    if (withDims) {
        return {
            logits: logitsForEvent,
            dims: logitsShape
        };
    } else {
        return logitsForEvent;
    }
}

export const generate = async (idx, age, options = {}) => {
    const {
        maxNewTokens = 100,
        maxAge = 85 * 365.25,
        noRepeat = true,
        terminationTokens = [1269],
        ignoreTokens = []
    } = options;

    const maskTime = -10000;
    const finalMaxTokens = maxNewTokens === -1 ? 128 : maxNewTokens;

    let currentIdx = [...idx];
    let currentAge = [...age];

    for (let step = 0; step < finalMaxTokens; step++) {
        const logits = await getDelphiLogits(currentIdx, currentAge);
        console.log(logits);
        for (const ignoreToken of ignoreTokens) {
            if (ignoreToken < logits.length) {
                logits[ignoreToken] = -Infinity;
            }
        }

        if (noRepeat) {
            const fill = currentIdx.map((token) => (token === 1 ? 0 : token));
            for (const token of fill) {
                if (token > 0 && token < logits.length) {
                    logits[token] = -Infinity;
                }
            }
        }

        const expLogits = logits.map((x) => Math.exp(-x));
        const randomSamples = logits.map(() => Math.random());
        const tSamples = expLogits.map((expLogit, i) =>
            Math.max(0, Math.min(365 * 80, -expLogit * Math.log(randomSamples[i])))
        );

        let minTime = Infinity;
        let minIndex = 0;
        for (let i = 0; i < tSamples.length; i++) {
            if (tSamples[i] < minTime) {
                minTime = tSamples[i];
                minIndex = i;
            }
        }

        const ageNext = currentAge[currentAge.length - 1] + minTime;
        currentIdx.push(minIndex);
        currentAge.push(ageNext);

        const hasTerminationToken = currentIdx.some((token) =>
            terminationTokens.includes(token)
        );
        const exceedsMaxAge = ageNext > maxAge;

        if (hasTerminationToken || exceedsMaxAge) {
            break;
        }
    }

    const pad = Array(currentIdx.length).fill(false);
    let terminationFound = false;
    let terminationCount = 0;

    for (let i = 0; i < currentIdx.length; i++) {
        if (terminationTokens.includes(currentIdx[i])) {
            terminationFound = true;
            terminationCount++;
        }

        if (terminationFound && terminationCount > 1) {
            pad[i] = true;
        }

        if (currentAge[i] > maxAge) {
            pad[i] = true;
        }
    }

    const finalLogitsOutput = await getDelphiLogits(
        currentIdx,
        currentAge,
        -1,
        true
    );
    for (let i = 0; i < pad.length; i++) {
        if (pad[i]) {
            currentIdx[i] = 0;
            currentAge[i] = maskTime;
        }
    }

    let processedLogits = [];
    const vocabSize = finalLogitsOutput.dims[2];
    if (noRepeat) {
        for (let seqPos = 0; seqPos < currentIdx.length; seqPos++) {
            const startIdx = seqPos * vocabSize;
            const positionLogits = finalLogitsOutput.logits.slice(
                startIdx,
                startIdx + vocabSize
            );

            const fill = currentIdx
                .slice(0, seqPos + 1)
                .map((token) => (token === 1 ? 0 : token));
            for (const token of fill) {
                if (token > 0 && token < positionLogits.length) {
                    positionLogits[token] = -Infinity;
                }
            }
            processedLogits.push(positionLogits);
        }
    } else {
        for (let seqPos = 0; seqPos < currentIdx.length; seqPos++) {
            const startIdx = seqPos * vocabSize;
            const positionLogits = finalLogitsOutput.logits.slice(
                startIdx,
                startIdx + vocabSize
            );
            processedLogits.push(positionLogits);
        }
    }

    return {
        idx: currentIdx,
        age: currentAge,
        logits: processedLogits
    };
}