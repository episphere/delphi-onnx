import { InferenceSession, Tensor } from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.mjs"

const MODEL_URL = "https://episphere.github.io/delphi-onnx/delphi.onnx"
const NUM_DAYS_IN_A_YEAR = 365.25
let instance = undefined

const mulberry32RNG = (seed = Date.now()) => {
    return function () {
        seed += 0x6D2B79F5
        let z = seed
        z = Math.imul(z ^ z >>> 15, z | 1)
        z ^= z + Math.imul(z ^ z >>> 7, z | 61)
        return ((z ^ z >>> 14) >>> 0) / 4294967296
    }
}

const fetchLabels = async () => {
    const baseURL = import.meta.url.split("/").slice(0, -1).join("/")
    const delphiLabelsURL = `${baseURL}/delphi_labels_chapters_colours_icd.json`
    return (await fetch(delphiLabelsURL)).json()
}

export default class DelphiONNX {
    constructor(options = {}) {
        const {
            modelURL = MODEL_URL,
            seed = Date.now()
        } = options
        this.modelURL = modelURL
        this.seed = seed
        this.rng = mulberry32RNG(this.seed)
        
        this.nameToTokenId = undefined
        this.tokenIdToName = undefined
    }

    async initialize() {
        
        this.nameToTokenId = {}
        this.tokenIdToName = {}

        if (!this.session) {
            await this.getModel()
        }      
        const delphiLabels = await fetchLabels()
        for (const obj of delphiLabels) {
            this.nameToTokenId[obj["name"]] = parseInt(obj["index"])
            this.tokenIdToName[obj["index"]] = obj["name"]
        }
    }

    async getModel() {
        this.session = await InferenceSession.create("https://episphere.github.io/delphi-onnx/delphi.onnx", {
            executionProviders: ["wasm", "cpu"]
        })
    }

    getTokensFromEvents(events = []) {
        let tokens = undefined
        if (Array.isArray(events)) {
            tokens = events.map(event => this.nameToTokenId[event])
        } else {
            tokens = this.nameToTokenId[events]
        }
        return tokens
    }

    getEventsFromTokens(tokens = []) {
        let events = undefined
        if (Array.isArray(tokens)) {
            events = tokens.map(tokenId => this.tokenIdToName[tokenId])
        } else {
            events = this.tokenIdToName[tokens]
        }
        return events
    }

    convertAgeToDays(ages = []) {
        let agesInDays = undefined
        if (Array.isArray(ages)) {
            agesInDays = ages.map(ageInYrs => ageInYrs * NUM_DAYS_IN_A_YEAR)
        } else {
            agesInDays = ages * NUM_DAYS_IN_A_YEAR
        }
        return agesInDays
    }

    convertAgeToYears(ages = [], precision = 1) {
        let agesInYears = undefined
        if (Array.isArray(ages)) {
            agesInYears = ages.map(ageInDays => (ageInDays / NUM_DAYS_IN_A_YEAR).toFixed(precision))
        } else {
            agesInYears = (ages / NUM_DAYS_IN_A_YEAR).toFixed(precision)
        }
        return agesInYears
    }

    getNextLogits(eventTokens, ages) {
        return this.getLogits(eventTokens, ages, eventTokens.length - 1)
    }
    
    getAllLogits(eventTokens, ages) {
        return this.getLogits(eventTokens, ages, - 1)
    }

    async getLogits(eventTokens, ages, logitsIndex) {
        if (!this.session || !this.rng) {
            await this.fetchModel()
        }
        if (!logitsIndex || !Number.isInteger(logitsIndex)) {
            logitsIndex = eventTokens.length - 1
        }

        // Only works for a batch size of 1!
        const idxTensor = new Tensor(
            "int64",
            new BigInt64Array(eventTokens.map((x) => BigInt(x))),
            [1, eventTokens.length]
        )

        const ageTensor = new Tensor("float32", new Float32Array(ages), [
            1,
            ages.length
        ])

        const feeds = {
            idx: idxTensor,
            age: ageTensor
        }

        const results = await this.session.run(feeds)
        let { data: logits, dims: logitsShape } = results.logits

        // const batchSize = logitsShape[0]; // Assuming batchSize always 1 for now, index calculation below will need adjustment otherwise
        const numEvents = logitsShape[1]
        const vocabSize = logitsShape[2]

        let logitsForEvent = undefined
        let dims = undefined
        if (Number.isInteger(logitsIndex) && logitsIndex >= 0 && logitsIndex < numEvents) {
            const eventLogitsIndex = logitsIndex * vocabSize
            logitsForEvent = logits.slice(
                eventLogitsIndex,
                eventLogitsIndex + vocabSize
            )
            dims = [1, vocabSize]
        } else if (logitsIndex === -1) {
            logitsForEvent = logits
            dims = logitsShape
        }

        return {
            logits: logitsForEvent,
            dims
        }
    }

    async generateTrajectory(idx, ages, options = {}) {
        const {
            maxNewTokens = 100,
            maxAge = 85 * NUM_DAYS_IN_A_YEAR,
            noRepeat = true,
            terminationTokens = [1269],
            ignoreTokens = []
        } = options

        const maskTime = -10000
        const finalMaxTokens = maxNewTokens === -1 ? 128 : maxNewTokens
        
        let currentIdx = [...idx]
        let currentAge = [...ages]

        for (let step = 0; step < finalMaxTokens; step++) {
            const { logits } = await this.getLogits(currentIdx, currentAge)

            ignoreTokens.forEach(ignoreToken => {
                if (ignoreToken < logits.length) {
                    logits[ignoreToken] = -Infinity
                }
            })

            if (noRepeat) {
                currentIdx.forEach(token => {
                    if (token > 1 && token < logits.length) {
                        logits[token] = -Infinity;
                    }
                })
            }

            const tSamples = logits.map(logit => {
                const randomSample = -Math.exp(-logit) * Math.log(this.rng())
                return Math.max(0, Math.min(maxAge, randomSample))
            })

            let minTime = Infinity
            let minIndex = -1
            tSamples.forEach((tSample, i) => {
                if (tSample < minTime) {
                    minTime = tSample;
                    minIndex = i;
                }
            })

            const ageNext = currentAge[currentAge.length - 1] + minTime
            currentIdx.push(minIndex)
            currentAge.push(ageNext)

            const hasTerminationToken = currentIdx.some((token) =>
                terminationTokens.includes(token)
            )
            const exceedsMaxAge = ageNext > maxAge;
            if (hasTerminationToken || exceedsMaxAge) {
                break
            }
        }

        const finalLogitsOutput = await this.getLogits(currentIdx, currentAge, -1)

        let terminationFound = false
        let terminationCount = 0
        currentIdx.forEach((event, i) => {
            if (terminationTokens.includes(event)) {
                terminationFound = true
                terminationCount++
            }
            if ((terminationFound && terminationCount > 1) || currentAge[i] > maxAge) {
                currentIdx[i] = 0
                currentAge[i] = maskTime
            }
        })

        let processedLogits = []
        const vocabSize = finalLogitsOutput.dims[2]
        if (noRepeat) {
            for (let seqPos = 0; seqPos < currentIdx.length; seqPos++) {
                const startIdx = seqPos * vocabSize;
                const positionLogits = finalLogitsOutput.logits.slice(
                    startIdx,
                    startIdx + vocabSize
                )

                currentIdx
                    .slice(0, seqPos + 1)
                    .forEach(token => {
                    if (token > 1 && token < positionLogits.length) {
                        positionLogits[token] = -Infinity
                    }
                })

                processedLogits.push(positionLogits)
            }
        } else {
            for (let seqPos = 0; seqPos < currentIdx.length; seqPos++) {
                const startIdx = seqPos * vocabSize;
                const positionLogits = finalLogitsOutput.logits.slice(
                    startIdx,
                    startIdx + vocabSize
                );
                processedLogits.push(positionLogits)
            }
        }

        return {
            tokenIds: currentIdx,
            age: currentAge,
            logits: processedLogits
        }
    }
}

export const generateTrajectory = async ({modelURL=MODEL_URL, seed=Date.now(), eventsList, options}) => {
    if (typeof(instance) === 'undefined') {
        instance = new DelphiONNX({modelURL, seed})
    }

    if (!Array.isArray(eventsList)) {
            throw new Error("Events List must be an array of objects!")
        }

        if (typeof(instance.nameToTokenId) === 'undefined') {
            await instance.initialize()
        }
        
        let idx = eventsList.map(e => e['event'])
        let ages = eventsList.map(e => e['age'])
        
        if (idx.some(evt => typeof(evt) === 'string')) {
            idx = instance.getTokensFromEvents(idx)
        }
        if (ages.every(ageForEvt => ageForEvt < 365)) {
            ages = instance.convertAgeToDays(ages)
        }

        const generatedTrajectory = await instance.generateTrajectory(idx, ages, options)
        const predictedEvents = instance.getEventsFromTokens(generatedTrajectory.tokenIds);
        const predictedAges = instance.convertAgeToYears(generatedTrajectory.age, 3);
        
        const reformattedTrajectory = predictedEvents.map((event, i) => {
            const obj = {
                event,
                age: predictedAges[i],
                logits: generatedTrajectory.logits[i]
            }
            return obj
        })
        
        return reformattedTrajectory
}