using System.Collections;
using System.Collections.Generic;
using System.Security.Cryptography;
using UnityEngine;
using System.Linq;
using System;
using System.Data;
using UnityEditor;
using Random = UnityEngine.Random;
using Unity.VisualScripting;

public class GameManager : MonoBehaviour
{
    public GameObject nn;
    public GameObject diagram;
    public List<GameObject> AIs = new List<GameObject>();
    public int changeWord = 0;
    public int amountOfRightAnswers = 0;
    public int factorN = 0;
    
    public List<int> oInn = new List<int>();
    public List<int> iInn = new List<int>();
    public List<bool> RNN = new List<bool>();
    public List<List<GameObject>> classAIs = new List<List<GameObject>>();
    public List<float> numberClassAIs = new List<float>();

    public int population = 1000;
    public int selection = 50;

    public int epoch = 0;
    public float rotationY = 0f;
    public int countBatch = 0;

    public int batchSize = 0;
    public Vector3 VectorDistance;

    public List<string> wordTranslate;
    public string englishPhrase;
    public string targetPhrase;

    public List<string> theBest = new List<string>();
    
    // Токенизатор
    private Tokenizer tokenizer;
    private int vocabSize;

    private HashSet<int> availableTokens = new HashSet<int>();
    private int currentPhase = 1;
    private int phrasesPerPhase = 1;
    private int initialPhrases = 5;
    public int phrasesIncrement = 1;
    public float phaseThreshold = 0.7f;

    void Start()
    {
        Phrases phrasesComponent = GetComponent<Phrases>();
        if (phrasesComponent == null)
        {
            Debug.LogError("Phrases component not found!");
            return;
        }
        
        var allPhrases = phrasesComponent.GetAllPhrases();
        
        if (allPhrases.Count == 0)
        {
            Debug.LogError("No phrases loaded!");
            return;
        }
        
        List<string> englishSentences = new List<string>();
        
        foreach (var phrasePair in allPhrases)
        {
            if (phrasePair.Count >= 2)
            {
                englishSentences.Add(phrasePair[1]);
            }
        }
        
        tokenizer = new Tokenizer(englishSentences);
        InitializeAvailableTokens();
        vocabSize = tokenizer.GetVocabSize();
        
        Debug.Log($"Tokenizer created: {vocabSize} English tokens");

        wordTranslate = phrasesComponent.GetPhrase();
        if (wordTranslate.Count >= 2)
        {
            englishPhrase = wordTranslate[1];
            targetPhrase = wordTranslate[1];
        }
        else
        {
            englishPhrase = "";
            targetPhrase = "";
        }
        bool ifNew = wordTranslate.Count >= 3 && wordTranslate[2] == "Yes";
        
        List<int> names = new List<int>();
        for (int i = 0; i < 100; i++)
        {
            int name = Random.Range(100, 501);
            while (names.IndexOf(name) != -1)
            {
                name = Random.Range(100, 501);
            }
            names.Add(name);
        }
        
        Time.timeScale = 15;
        VectorDistance = gameObject.transform.position;
        
        for (int i = 0; i < population; i++)
        {
            GameObject a = Instantiate(nn, VectorDistance, Quaternion.identity);
            
            VectorDistance.x += 1.5f;
            if (VectorDistance.x >= transform.position.x + 12f)
            {
                VectorDistance.x = transform.position.x;
                VectorDistance.z += 3;
                VectorDistance.x += 1;
            }
            
            a.transform.Rotate(0f, rotationY, 0f);
            
            AI aiComponent = a.GetComponent<AI>();
            aiComponent.spawnerOfNN = gameObject;
            aiComponent.addition = 1;
            
            aiComponent.SetTokenizer(tokenizer);
            aiComponent.UpdateAvailableTokens(availableTokens);
            
            aiComponent.inputPhrase = englishPhrase;
            aiComponent.targetPhrase = targetPhrase;
            aiComponent.ifNew = ifNew;
            aiComponent.inputTokens = tokenizer.Tokenize(englishPhrase);
            aiComponent.targetTokens = tokenizer.Tokenize(targetPhrase);
            
            a.name = Random.Range(1, 10000).ToString();
            AIs.Add(a);
        }
        
        VectorDistance = transform.position;
    }

    void Update()
    {
        bool startNewEpoch = true;
        foreach (var a in GameObject.FindGameObjectsWithTag("Player"))
        {
            if (a.GetComponent<AI>().go != false)
            {
                startNewEpoch = false;
            }
        }
        
        if (startNewEpoch)
        {
            Phrases phrasesComponent = GetComponent<Phrases>();
            wordTranslate = phrasesComponent.GetPhrase();
            
            if (wordTranslate.Count >= 2)
            {
                englishPhrase = wordTranslate[1];
                targetPhrase = wordTranslate[1];
            }
            bool ifNew = wordTranslate.Count >= 3 && wordTranslate[2] == "Yes";
            
            AIs.Clear();
            
            if (countBatch < batchSize)
            {
                List<GameObject> listOfAIs = GameObject.FindGameObjectsWithTag("Player")
                .Where(o => o.GetComponent<AI>().spawnerOfNN.name == gameObject.name)
                .OrderByDescending(o => o.GetComponent<AI>().howIsGood).ToList();
                AI _bestAI = listOfAIs[0].GetComponent<AI>();
                if (_bestAI.ifNew == true && _bestAI.textHowIsGood > phaseThreshold)
                {
                    ++amountOfRightAnswers;
                }

                foreach (var a in listOfAIs)
                {
                        float tempGood = a.GetComponent<AI>().howIsGood;
                        AI ai = a.GetComponent<AI>();
                        
                        ai.inputPhrase = englishPhrase;
                        ai.targetPhrase = targetPhrase;
                        ai.ifNew = ifNew;

                        ai.inputTokens = tokenizer.Tokenize(englishPhrase);
                        ai.targetTokens = tokenizer.Tokenize(targetPhrase);
                        
                        ai.Start();
                }
                
                ++countBatch;
                return;
            }
            
            countBatch = 0;
            
            foreach (var a in GameObject.FindGameObjectsWithTag("Player"))
            {
                AI ai = a.GetComponent<AI>();
                ai.howIsGood = ai.GetBatchAverageFitness();
                ai.ResetBatchStats();
            }

            foreach (var a in GameObject.FindGameObjectsWithTag("Player"))
            {
                AIs.Add(a);
            }
            
            List<int> names = new List<int>();
            for (int i = 0; i < 100; i++)
            {
                int name = Random.Range(100, 701);
                while (names.IndexOf(name) != -1)
                {
                    name = Random.Range(100, 701);
                }
                names.Add(name);
            }
            
            classAIs.Clear();
            List<GameObject> copyAI = new List<GameObject>();
            factorN = 0;
            
            foreach (GameObject a in AIs)
            {
                copyAI.Add(a);
                if (factorN < a.GetComponent<AI>().inpInnov.Count)
                {
                    factorN = a.GetComponent<AI>().inpInnov.Count;
                }
            }
            
            List<GameObject> SortedList = new List<GameObject>();
            {
                int i = 0;
                List<GameObject> b = AIs.OrderByDescending(o => o.GetComponent<AI>().howIsGood).ToList();

                diagram.GetComponent<Diagram>().AddFitnessValue(b[0].GetComponent<AI>().howIsGood);
                diagram.GetComponent<Diagram>().AddWordsValue(b[0].GetComponent<AI>().textHowIsGood);
                diagram.GetComponent<Diagram>().AddVocabularyValue(phrasesComponent.lengthOfknown);

                foreach (var a in b)
                {
                    if (i >= selection - 15)
                    {
                        break;
                    }
                    ++i;
                    SortedList.Add(a);
                }
                
                for (int j = 0; j != 15; ++j)
                {
                    SortedList.Add(b[Random.Range(selection - 15 + 1, b.Count)]);
                }
            }
            AI bestAI = SortedList[0].GetComponent<AI>();
            string generatedText = bestAI.generatedText;
            if (string.IsNullOrEmpty(generatedText) && tokenizer != null && bestAI.outputTokens.Count > 0)
            {
                generatedText = tokenizer.Detokenize(bestAI.outputTokens);
            }

            string logEntry = $"[Epoch {epoch}] Fitness: {bestAI.textHowIsGood:F4} | " +
                            $"Target: {bestAI.targetPhrase} | " +
                            $"Generated: {generatedText}";
            theBest.Add(logEntry);
            
            if(bestAI.ifNew == true && bestAI.textHowIsGood >= phaseThreshold)
            {
                ++amountOfRightAnswers;
            }

            if (amountOfRightAnswers >= 3)
            {
                var allPhrases = GetComponent<Phrases>().GetAllPhrases();

                int startIdx = currentPhase;
                int endIdx = Mathf.Min(startIdx + phrasesPerPhase, allPhrases.Count);

                for (int i = startIdx; i != endIdx; ++i)
                {
                    var enTokens = tokenizer.Tokenize(allPhrases[i][1]);

                    foreach (var token in enTokens){
                        if (token >= 4) availableTokens.Add(token);
                    }
                }

                currentPhase = Mathf.Min(currentPhase + phrasesPerPhase, Mathf.CeilToInt((float)allPhrases.Count / phrasesPerPhase) - 1);

                Debug.Log($"Advanced to Phase {currentPhase}: {availableTokens.Count} tokens available");

                foreach (var aiObj in GameObject.FindGameObjectsWithTag("Player"))
                {
                    AI ai = aiObj.GetComponent<AI>();
                    ai.UpdateAvailableTokens(availableTokens);
                    ai.ResetForNewPhase();
                }

                phrasesComponent.changeWord = true;
                amountOfRightAnswers = 0;
                ifNew = false;
            }


            List<GameObject> NewAIs = new List<GameObject>();
            
            classAIs.Add(new List<GameObject>());
            classAIs[0].Add(SortedList[0]);
            
            for (int k = 0; k < SortedList.Count; k++)
            {
                GameObject thisNN = SortedList[k];
                List<int> thisNNInnvoations = new List<int>(thisNN.GetComponent<AI>().innovations);
                thisNNInnvoations.Sort();
                
                float fit = 0;
                float disjoint = 0;
                float excess = 0;
                float differenceWeight = 0;
                int differenceWeightLength = 0;
                
                for (int i = 0; i < classAIs.Count; i++)
                {
                    GameObject c = classAIs[i][0];
                    List<int> cInnvoations = new List<int>(c.GetComponent<AI>().innovations);
                    cInnvoations.Sort();
                    
                    int countInnov = cInnvoations.Count - 1;
                    for (int j = 0; j < thisNNInnvoations.Count; j++)
                    {
                        if (j > countInnov)
                        {
                            excess += countInnov + 1;
                            break;
                        }
                        else
                        {
                            if (cInnvoations.IndexOf(thisNNInnvoations[j]) == -1)
                            {
                                ++disjoint;
                            }
                            else
                            {
                                differenceWeight += Mathf.Abs(thisNNInnvoations[j] - cInnvoations[cInnvoations.IndexOf(thisNNInnvoations[j])]);
                                ++differenceWeightLength;
                            }
                        }
                    }
                    
                    fit = (disjoint + excess) / factorN + differenceWeight / differenceWeightLength;
                    
                    if (fit > 0.8)
                    {
                        classAIs[i].Add(SortedList[k]);
                        break;
                    }
                    else if (i == (classAIs.Count - 1))
                    {
                        classAIs.Add(new List<GameObject>());
                        classAIs[classAIs.Count - 1].Add(SortedList[k]);
                        break;
                    }
                }
            }

            for (int i = 0; i < population - selection; i++)
            {
                int FirstInd = Random.Range(0, classAIs.Count);
                int copyAICount = copyAI.Count;
                GameObject parent1 = classAIs[FirstInd][Random.Range(0, classAIs[FirstInd].Count)];
                GameObject parent2 = classAIs[FirstInd][Random.Range(0, classAIs[FirstInd].Count)];

                if (parent2.GetComponent<AI>().howIsGood > parent1.GetComponent<AI>().howIsGood)
                {
                    (parent1, parent2) = (parent2, parent1);
                }

                GameObject offspring = Instantiate(parent1, 
                    copyAI[Random.Range(0, copyAICount)].transform.position + 
                    new Vector3(Random.Range(0.5f, 1.5f), 0f, Random.Range(0.5f, 1.5f)), 
                    Quaternion.identity);
                
                AI offspringAI = offspring.GetComponent<AI>();
                offspringAI.SetTokenizer(tokenizer);
                offspringAI.UpdateAvailableTokens(availableTokens);

                offspringAI.inputPhrase = englishPhrase;
                offspringAI.targetPhrase = targetPhrase;
                offspringAI.ifNew = ifNew;
                offspringAI.inputTokens = tokenizer.Tokenize(englishPhrase);
                offspringAI.targetTokens = tokenizer.Tokenize(targetPhrase);
                offspringAI.spawnerOfNN = gameObject;
                offspringAI.Start();
                
                if (parent2.GetComponent<AI>().howIsGood == parent1.GetComponent<AI>().howIsGood)
                {
                    if (parent2.GetComponent<AI>().innovations.Any())
                    {
                        for (int j = 0; j < parent2.GetComponent<AI>().innovations.Count; j++)
                        {
                            int tryFind = offspringAI.innovations.IndexOf(parent2.GetComponent<AI>().innovations[j]);
                            if (tryFind == -1)
                            {
                                offspringAI.inpInnov.Add(parent2.GetComponent<AI>().inpInnov[j]);
                                offspringAI.outInnov.Add(parent2.GetComponent<AI>().outInnov[j]);
                                offspringAI.weights.Add(parent2.GetComponent<AI>().weights[j]);
                                offspringAI.innovations.Add(parent2.GetComponent<AI>().innovations[j]);
                                offspringAI.actConnect.Add(parent2.GetComponent<AI>().actConnect[j]);
                                offspringAI.RNNs.Add(parent2.GetComponent<AI>().RNNs[j]);

                                List<float> InitalAmountOfNeurones = new List<float>(offspringAI.neurones);
                                
                                if (parent2.GetComponent<AI>().inpInnov[j] + 1 > offspringAI.neurones.Count)
                                {
                                    for (int k = 0; k < parent2.GetComponent<AI>().inpInnov[j] + 1 - offspringAI.neurones.Count; k++)
                                    {
                                        offspringAI.neurones.Add(0f);
                                        offspringAI.RNNneurones.Add(0f);
                                    }
                                }
                                
                                if (parent2.GetComponent<AI>().outInnov[j] + 1 > offspringAI.neurones.Count)
                                {
                                    for (int k = 0; k < parent2.GetComponent<AI>().outInnov[j] + 1 - offspringAI.neurones.Count; k++)
                                    {
                                        offspringAI.neurones.Add(0f);
                                        offspringAI.RNNneurones.Add(0f);
                                    }
                                }
                                
                                try
                                {
                                    if (offspringAI.GenToPh().SequenceEqual(new List<int> { 1, offspringAI.getOutConnections(), 0 }))
                                    {
                                        int amountOfInnov = offspringAI.inpInnov.Count - 1;

                                        offspringAI.inpInnov.RemoveAt(amountOfInnov);
                                        offspringAI.outInnov.RemoveAt(amountOfInnov);
                                        offspringAI.weights.RemoveAt(amountOfInnov);
                                        offspringAI.innovations.RemoveAt(amountOfInnov);
                                        offspringAI.actConnect.RemoveAt(amountOfInnov);
                                        offspringAI.RNNs.RemoveAt(amountOfInnov);
                                        offspringAI.neurones = new List<float>(InitalAmountOfNeurones);
                                        offspringAI.RNNneurones = new List<float>(InitalAmountOfNeurones);
                                    }
                                }
                                catch
                                {
                                    Debug.Log(offspringAI.neurones.Count.ToString() + " " + 
                                              parent2.GetComponent<AI>().inpInnov[j].ToString() + " " + 
                                              parent2.GetComponent<AI>().outInnov[j].ToString());
                                }
                            }
                        }
                        offspringAI.Start();
                    }
                }

                if (offspringAI.innovations.Any())
                {
                    if (parent2.GetComponent<AI>().innovations.Any())
                    {
                        for (int j = 0; j < offspringAI.innovations.Count; j++)
                        {
                            if (Random.Range(0, 2) == 0)
                            {
                                int tryFind = parent2.GetComponent<AI>().innovations.IndexOf(offspringAI.innovations[j]);
                                if (tryFind != -1)
                                {
                                    offspringAI.weights[j] = parent2.GetComponent<AI>().weights[tryFind];
                                }
                            }
                        }
                    }
                }
                
                int chooseAddition = Random.Range(0, 10);
                if (chooseAddition < 4)
                {
                    offspringAI.addition = 1;
                }
                else if (chooseAddition < 6)
                {
                    offspringAI.addition = 2;
                }
                else
                {
                    offspringAI.addition = 0;
                }
                
                if (offspringAI.actConnect.Any())
                {
                    if (Random.Range(0, 5) < 1)
                    {
                        offspringAI.actConnect[Random.Range(0, offspringAI.actConnect.Count)] = false;
                    }
                    else if (Random.Range(0, 5) < 1)
                    {
                        List<int> check = new List<int>();
                        for (int j = 0; j < offspringAI.actConnect.Count; j++)
                        {
                            if (offspringAI.actConnect[j] == false)
                            {
                                check.Add(j);
                            }
                        }
                        if (check.Any())
                        {
                            int reactivate = Random.Range(0, check.Count);
                            offspringAI.actConnect[check[reactivate]] = true;
                            if (offspringAI.correctGen(offspringAI.GenToPh()) == true)
                            {
                                offspringAI.RNNs[check[reactivate]] = true;
                            }
                        }
                    }
                }
                
                for (int ie = 0; ie < offspringAI.weights.Count; ie++)
                {
                    float r = Random.Range(0f, 1f);
                    if (r < 0.1f)
                    {
                        offspringAI.weights[ie] += Random.Range(-1f, 1f);
                    }
                    else if (r < 0.3f)
                    {
                        offspringAI.weights[ie] += Random.Range(-0.1f, 0.1f);
                    }
                }
                
                NewAIs.Add(offspring);
                
                VectorDistance.x += 1.5f;
                if (VectorDistance.x >= transform.position.x + 12f && VectorDistance.z <= 6 + transform.position.z)
                {
                    VectorDistance.x = transform.position.x;
                    VectorDistance.z += 4f;
                    VectorDistance.x += 1.5f;
                }
                else if (VectorDistance.x >= transform.position.x + 12f)
                {
                    VectorDistance.x = transform.position.x;
                    VectorDistance.z -= 8f;
                    VectorDistance.x -= 3f;
                }
                
                offspring.transform.Rotate(0f, rotationY, 0f);
                
                try
                {
                    offspring.name = names[i].ToString();
                }
                catch
                {
                    offspring.name = Random.Range(1000, 10000).ToString();
                }
                
                offspringAI.inputPhrase = englishPhrase;
                offspringAI.targetPhrase = targetPhrase;
                offspringAI.ifNew = ifNew;
                offspringAI.spawnerOfNN = gameObject;
                offspringAI.Start();
            }
            
            wordTranslate = phrasesComponent.GetPhrase();

            AIs = AIs.OrderByDescending(o => o.GetComponent<AI>().howIsGood).ToList();
            for (int i = 0; i != population; ++i)
            {
                if (population != AIs.Count)
                {
                    Debug.Log("Not equal amount!");
                }
                if (i < 50)
                {
                    SortedList[i].GetComponent<AI>().inputPhrase = englishPhrase;
                    SortedList[i].GetComponent<AI>().targetPhrase = targetPhrase;
                    SortedList[i].GetComponent<AI>().ifNew = ifNew;
                    SortedList[i].GetComponent<AI>().inputTokens = tokenizer.Tokenize(englishPhrase);
                    SortedList[i].GetComponent<AI>().targetTokens = tokenizer.Tokenize(targetPhrase);
                    SortedList[i].GetComponent<AI>().Start();
                    SortedList[i].name = "Prev";
                }
                else
                {
                    Destroy(AIs[i]);
                }
            }
            
            ++epoch;
        }
    }

    private void InitializeAvailableTokens()
    {
        availableTokens.Clear();
        
        // Всегда добавляем служебные токены
        availableTokens.Add(Tokenizer.PAD_TOKEN);
        availableTokens.Add(Tokenizer.SOS_TOKEN);
        availableTokens.Add(Tokenizer.EOS_TOKEN);
        availableTokens.Add(Tokenizer.UNK_TOKEN);
        
        var allPhrases = GetComponent<Phrases>().GetAllPhrases();
        int phrasesToUse = Mathf.Min(initialPhrases, allPhrases.Count);
        
        Debug.Log($"Initializing tokens: {phrasesToUse} phrases to use from {allPhrases.Count} total");
        
        for (int i = 0; i < phrasesToUse; i++)
        {
            var tokens = tokenizer.Tokenize(allPhrases[i][1]); // Используем английскую часть
            
            foreach (var token in tokens)
            {
                if (token >= 4) // Исключаем служебные
                {
                    availableTokens.Add(token);
                }
            }
        }
        
        GetComponent<Phrases>().lengthOfknown = phrasesToUse;
        currentPhase = phrasesToUse;
        Debug.Log($"Phase {currentPhase}: {availableTokens.Count} tokens available");
        
        // Логгируем первые несколько токенов для отладки
        var firstTokens = availableTokens.Take(20).ToList();
        Debug.Log($"First 20 available tokens: {string.Join(", ", firstTokens)}");
    }
    
    public int DealInnovations(int inoV, int outV, bool rnnV)
    {   
        for (int i = 0; i < iInn.Count; i++)
        {
            if (iInn[i] == inoV && oInn[i] == outV && RNN[i] == rnnV)
            {
                return i;
            }
        }
        
        iInn.Add(inoV);
        oInn.Add(outV);
        RNN.Add(rnnV);
        return iInn.Count - 1;
    }
}