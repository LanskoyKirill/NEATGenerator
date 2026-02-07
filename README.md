# NEATGenerator
To write documentation to facilitate the entry of people unfamiliar with the project, a generative model was used (ChatGPT5.2)

**In short:** This is an experimental text generator written in Unity with an implementation of a simplified version of the NEAT (NeuroEvolution of Augmenting Topologies) algorithm. The project evolves a population of small neural networks (genomes) that sequentially generate tokenized sentences. The goal is for the networks to learn to reproduce given phrases (autocompression/autogeneration) – each phrase is both an input and a target sequence.

In this README, I will describe in detail the project architecture, key classes and methods, the training/mutation/crossover algorithm, data format, launch parameters, debugging guidelines, and improvement recommendations – so that any developer can quickly get started and continue working.

---

## Contents

1. [What's in the project](#what-in-the-project)
2. [How it works (general description)](#how-it-works-general-description)
3. [Class-by-class - functionality and key methods](#class-by-class---functionality-and-key-methods)
4. [NEAT implementation - algorithm details](#neat-implementation---algorithm-details)
5. [Fitness function (how quality is assessed)](#fitness-function-how-quality-is-assessed)
6. [Data format and tokenization](#data-format-and-tokenization)
7. [Startup, parameters, and settings](#startup-parameters-and-settings)
8. [Debugging: typical problems and how to fix them] [find](#debugging-typical-problems-and-how-to-find-them)
9. [Improvement and Scaling Recommendations](#improvement-and-scaling-recommendations)
10. [TODOs and Known Issues](#todos-and-known-issues)
11. [License and Contributions](#license-and-contributions)

---

## What's in the Project

Files (main):

* `GameManager.cs` — orchestrator: loading phrases, creating a population, managing epochs/batches, selection, creating offspring, managing phases (progressive dictionary expansion).
* `AI.cs` — individual object: contains the genome (list of connections/neurons/innovations), sequence generation logic, mutations (AddLink/AddNode), calculation ordering, RNN support, and fitness calculation.
* `Phrases.cs` — a simple phrase pool (a list of English sentences) and the logic for generating pairs (input/target); stores `lengthOfknown` — how many phrases are already "known" to the system.
* `Tokenizer.cs` — simple tokenization: builds a dictionary from the passed sentences, supports special tokens (`PAD`, `SOS`, `EOS`, `UNK`), and the `Tokenize` / `Detokenize` methods.

--

## How it works (general description)

1. `GameManager` collects a list of phrases via `Phrases.GetAllPhrases()` and creates a `Tokenizer`.
2. At startup, a `population` of neural network prefab instances (`nn`) is created, each with a copied set of parameters (random names, material, initial structures).
3. Each individual (AI) generates a tokenized sequence in its Update() function, like an autoregressive model, using the previous token (starting with SOS) as input.
4. After generation is complete (EOS, iteration or length limit), the fitness of the individual is calculated.
5. GameManager aggregates fitnesses (by batch), selects the best (selection + some random inclusion), clusters individuals (by innovations), and creates descendants within classes (crossover + inheritance and additions).
6. Mutation: changing weights, enabling/disabling links, adding a link (AddLink), and adding a node (AddNode) – all implemented manually with the innovation tracker (DealInnovations in GameManager). 7. Phases: As the system "manages" to produce correct phrases (`amountOfRightAnswers`), `GameManager` expands additional tokens/words into `availableTokens`, increasing the complexity.

---

## Class-by-class — functionality and key methods

### `GameManager` (order)

**Main purpose:** initialization, epoch control, selection, and breeding.

Key fields (important for configuration):

* `public int population` — population size.
* `public int selection` — number of individuals selected as "elite" (and used in selection).
* `public int batchSize` — number of examples/episodes in one batch.
* `public float phaseThreshold` — quality threshold for moving to the next phase (increasing the available tokens).
* `public int phrasesIncrement`, `initialPhrases` — parameters for gradual dictionary expansion.

Key methods/blocks:

* `Start()` — creates a `Tokenizer`, initializes `availableTokens`, creates NN instances (`Instantiate(nn)`), assigns them `Tokenizer`, `inputPhrase`, `targetPhrase`, and initializes the starting tokens.
* `Update()` — main loop: waits for all networks in the epoch to complete; Then:

* Collects fitness (batch), aggregates, updates `howIsGood` for each individual,
* Sorts the population by fitness (`howIsGood`) and forms `SortedList` — top + randomly selected,
* Performs clustering by innovation structure (`classAIs`),
* Within a class, takes parents and generates children (copying the structure + inheriting innovations from the second parent with equal fitness),
* Apply mutations to weights/connection on/off,
* Updates positional coordinates in the scene and prepares a new epoch.
* `D`
* ealInnovations(int inoV, int outV, bool rnnV)` — a centralized innovation tracker. Returns the index of a unique pair (in, out, rnn) — used to synchronize innovations between individuals.

**Note:** `availableTokens` is a set of token indices that are allowed to be used in generations (training phases).

---

### `AI` (individual — neural network)

**Genome structure:**

* `neurones` — an array of numbers (the activation value of each neuron).
* `inpInnov`, `outInnov`, `weights`, `actConnect`, `RNNs`, `innovations` — parallel arrays representing connections (gene per connection): input neuron index, output neuron index, weight, whether the connection is active, whether the connection is recurrent, and a unique innovation ID. * `RNNneurones` — the state of the recurrent accumulators.
* `order`, `adjList` — auxiliary structures for topological order and adjacency lists.

**Key parameters:**

* `vocabSize` — the number of vocabulary tokens (set by `Tokenizer`).
* `outConnections` — the start index of the output neurons (bias + input tokens → outputs start after).
* `initalNeurones` — the default number of neurons required: `1 + vocabSize + vocabSize` (bias + inputs + outputs).

**Lifecycle:**

* `Start()`:

* Initializes neurons (bias, inputs, outputs), clears invalid connections (if indices are out of bounds), calls `UpdateAvailableNeurons()`, `makeOrder()`, and `Adder()`.
* If a `tokenizer` is present, converts the input/target phrases into `inputTokens` / `targetTokens`.
* `Update()`:

* Actual generation in `ProcessGenerationLogic()` only if `go == true`.
* If `inpInnov` is empty, the object is destroyed (invalid instance).
* `ProcessGenerationLogic()`:

* sets the input token (autoregressive), passes the signal through the neural network in order (activation `tanh`),
* selects the output token—the neuron with the maximum activation in the range of output neurons (`outConnections .. outConnections + vocabSize - 1`),
* updates `outputTokens`, `generatedText`, checks the termination condition (EOS, length, max iterations),
* calls `UpdateRNNState()` (update the recurrence buffer).
* `FinishGeneration()` → `CalculateTokenFitness()`: calculates `howIsGood` and `textHowIsGood`.

**Mutations/structural changes:**

* `AddLink()`—adds a new link (checking for existing duplicates and cycles). If unsuccessful, rolls back the changes and tries again (up to the limit). * `AddNode()` — splits an existing connection, creates a new neuron and two connections (similar to NEAT).
* `GenToPh()` — builds a local copy of the connections and attempts a topological sort (will return a special marker `{1, outConnections, 0}` if a cycle or mismatch is found).
* `makeOrder()` calls `GenToPh()` and stores the result in `order`.
* `Adder()` — calls `AddLink()` or `AddNode()` depending on the `addition` flag.

**Utility methods:**

* `UpdateAvailableNeurons()` — generates lists of available input and output neurons for mutations (takes into account bias, inputs, outputs, and hidden neurons).
* `GetBatchAverageFitness()`, `ResetBatchStats()` — support for batch fitness aggregation.

---

### `Phrases`

Stores a list of `phrases` strings. Returns `[input, target]` pairs (in the current implementation, this is the same phrase twice, meaning the task is to reconstruct the same phrase). The `GetPhrase()` logic monitors `lengthOfknown` and `changeWord` for gradual expansion of new phrases and the `ifNew` flag (whether a new phrase has been selected).

--

### `Tokenizer`

Simple dictionary tokenizer:

* Builds a dictionary based on the list of sentences in the constructor (`vocab`).
* Special tokens:
`PAD=0, SOS=1, EOS=2, UNK=3`
* `Tokenize(string)` — returns a list of tokens with `SOS` at the beginning and `EOS` at the end; unknown words → `UNK`.
* `Detokenize(List<int>)` — converts tokens back to a string (ignores `SOS`, `PAD`, stops on `EOS`).

Cleanup rules: retains letters, numbers, spaces, and apostrophes; replaces punctuation with spaces.

---

## NEAT Implementation — Algorithm Details

This project implements a simplified NEAT idea:

* **Genome:** is represented as a list of genes. Each gene is a tuple of `(inputNeuronIndex, outputNeuronIndex, weight, activeFlag, rnnFlag, innovationId)`.
* **Innovations:** `GameManager.DealInnovations` maintains a global table of `(in, out, rnn) -> innovationIndex` so that identical structural mutations receive the same innovation id (important for correct crossover/matching).
* **Crossover:** When creating an offspring, `parent1` (the best one) is copied, then, if the fitness is equal, `parent2` is scanned and missing genes from `parent2` are added (see the `if (parent2.howIsGood == parent1.howIsGood) ...` block).
* **Mutation:** Weights are slightly modified (randomly offset), there is a chance to turn on/off a connection (`actConnect`), and there is logic for randomly turning on `RNNs` for new connections.
* **Adding a node/link:** `AddNode()` and `AddLink()` implement structural mutations.
* **Specialization/classes (speciation-like):** The project groups individuals into `classAIs` based on different
* A combination of innovations and some metrics (`disjoint`, `excess`, `differenceWeight`). This is intended to preserve diversity; crossover occurs within a class.

---

## Fitness function (how quality is assessed)

`AI.CalculateTokenFitness()` combines several components:

1. **Exact Match Score** — the proportion of matching tokens at the same positions relative to the length of the target sequence (weight 0.4).
2. **Word Match Score** — a comparison of the set of unique (non-service) tokens between the output and the target (weight 0.3).
3. **LCS Score** — the normalized length of the longest common subsequence (weight 0.2).
4. **Structure Bonus** — a small bonus for having `SOS` and `EOS` in the right places (+0.03 each).
5. **Penalty For Length** — a penalty if the output is significantly longer than the target.
6. **Diversity Bonus** — a bonus for using different words (up to 0.1).
7. **Repetition Penalty** — a penalty for repeating the same word multiple times (more than 5 times).

The result is normalized and limited to a maximum value (200.0f in the code), then copied to `howIsGood` and accumulated for batch evaluation.

---

## Data Format and Tokenization

* `Phrases.phrases` — a simple list of strings (English sentences). Each string is used as input and as a target.
* `Tokenizer` cleans the text: leaves letters/numbers/apostrophes, replaces punctuation with spaces, and converts to lower case.
* During tokenization:

* `SOS` (1) is added to the beginning.
* Last — `EOS` (2).
* Unknown words — `UNK` (3).
* `vocabSize = tokenizer.GetVocabSize()` — the number of all tokens (including special tokens).

---

## Launch, Parameters, and Settings

### Requirements

* Unity (Unity project; tested on Unity 2020/2021 versions or later — the actual version depends on the project; make sure you have the required packages installed).
* The scene must contain `GameManager` (a link to the `nn` prefab), `diagram` (if needed), and a neural network prefab with an `AI` component.

### Quick Start

1. Open the project in Unity.
2. Drag `GameManager` onto an empty `GameObject` in the scene.
3. For initial debugging, reduce the parameters:

```csharp
GameManager.population = 100; // instead of 1000 — quick testing
GameManager.selection = 50;
GameManager.batchSize = 1;
Time.timeScale = 1;
```

4. Run the scene. The Unity console will log the steps and the best individuals (the `theBest` list is populated).

### Useful parameters (what to change)

* `population` — number of individuals. Larger values ​​→ slower but more stable evolution.
* `selection` — how much is saved/used during breeding.
* `batchSize` — how often to aggregate fitness data and switch between batches/epochs.
* `phaseThreshold`, `phrasesIncrement`, `initialPhrases` — control the gradual expansion of the vocabulary (`availableTokens`) and the complexity of the task.

---

## Debugging: Common Problems and How to Find Them

### 1. "Learning Stuck" — Sign

Symptoms: No fitness improvement in graphs/logs, individuals generate the same (or meaningless) output.

**Checks:**

* Ensure that `tokenizer` is created correctly and `vocabSize > 0`.
* Ensure that all agents have `outConnections == 1 + vocabSize` immediately after `SetTokenizer()` and after `Start()`: desynchronization of this value disrupts the indexing of output neurons. (This repository already has a fix: `SetTokenizer` sets `outConnections = 1 + vocabSize`.)
* Log the result of `GenToPh()`: if it returns the special token `{1, outConnections, 0}`, then topological sorting failed → the network contains a cycle/connectivity issue.
* Check that `targetTokens` is not empty (otherwise the network won't be able to compare anything).
* Check that `inpInnov`, `outInnov`, `weights`, `actConnect`, and `RNNs` are the same length (parallel arrays).
* Reduce `population` and `Time.timeScale`, run several epochs, and manually inspect the top few individuals (print `generatedText` and `outputTokens`).

### 2. Frequent `NullReference` / `IndexOutOfRange`

* Logs will help track `inputNeuronIndex`, `outputStart/outputEnd` in `ProcessGenerationLogic()`. Check that `neurones.Count` and `outConnections + vocabSize` match.
* If `GenToPh()` returns an incorrect order, print `adjList` and input/output connections to find the cause of the loop.

### 3. Performance

* `population` is too large: Unity will create/destroy thousands of GameObjects, which is expensive. To study the algorithm, it's better to work with a smaller population.
* If you want to speed things up, move the purely computational part (evolution, crossover, fitness) to non-Mono classes and execute them outside the render frame (coroutines / background worker).

---

## Recommendations for improvement and scaling

1. **Move calculations outside the GameObject loop.** Create "lightweight" objects (not GameObjects) for evolution—this way, 10,000 individuals won't weigh down the scene.
2. **Unit tests** for `GenToPh()`, `DealInnovations()`, `AddLink()`—this will reduce the number of bugs during modifications.
3. **More correct crossover**: currently, with equal fitness, simple gene merging is used—it's better to use innova matching.
4. tionId and handle disjoint/excess accurately.
4. **A richer fitness function**: can use n-gram, BLEU, edit distance, etc. metrics.
5. **Saving/Loading**: serialize the best specimen to JSON for further analysis.
6. **Abandoning `FindGameObjectsWithTag` in Update** - more efficient caching and collection handling.
7. **Add replay/visualization**: progress chart, network view, order/adjList visualization.

---

## TODO and Known Issues

* Fixed `outConnections` desynchronization in `SetTokenizer()` (previously, `outConnections` was set to 1, and then `1 + vocabSize` was set in Start - this could break output indexing).
* The code still contains operations that can lead to `IndexOutOfRange` for incorrect innovations. Carefully monitor the length of parallel lists (`inpInnov`, `outInnov`, `weights`, etc.).
* Clustering by innovations is implemented, but the `fit`/`0.8` threshold parameters were chosen empirically. This could be improved and made configurable.
* There is no persistence mechanism (for saving the best individuals) - this should be added.

---

## Examples of useful commands / code snippets

Change parameters in `GameManager` (recommended for debugging):

```csharp
public int population = 100; // was 1000 in the original
public int selection = 50;
public int batchSize = 1;
Time.timeScale = 1;
```

Printing a problematic topological sort (in `AI.GenToPh()`):

```csharp
if (order1.Count != neuronsCount) {
Debug.LogError($"Topological sort failed: neurons={neuronesCount}, orderCount={order1.Count}, outConnections={outConnections}");
return new List<int> { 1, outConnections, 0 };
}
```

Checking outConnections synchronization (in `GameManager`, immediately after `SetTokenizer`):

```csharp
foreach (var agent in AIs) {
var ai = agent.GetComponent<AI>();
if (ai.getOutConnections() != ai.vocabSize + 1) {
Debug.LogWarning($"outConnections desync: agent={agent.name}, outConnections={ai.getOutConnections()}, vocabSize={ai.vocabSize}");
}
}
```

