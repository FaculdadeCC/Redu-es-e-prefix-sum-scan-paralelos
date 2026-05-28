Uso de Inteligência Artificial
Este projeto contou com o auxílio de ferramentas de Inteligência Artificial
durante o processo de desenvolvimento, organização e documentação.
As IAs foram utilizadas principalmente para:

Esclarecimento de conceitos relacionados a CUDA, redução paralela e
prefix-sum (scan);
Apoio na estruturação do projeto e divisão de tarefas entre os integrantes;
Revisão e depuração de código;
Geração do notebook de benchmark completo (Fase 3) com medição via
cudaEvent, cálculo de speedup e exportação de gráficos/CSVs;
Redação das seções de Metodologia, Resultados e Discussão, e Conclusões
do relatório científico no template SBC;
Geração do roteiro de slides e estrutura da apresentação oral;
Auxílio na documentação em Markdown e LaTeX;
Sugestões de otimização e boas práticas de programação paralela.

Todo o código foi estudado, adaptado e validado manualmente antes de sua
utilização no projeto. Qualquer integrante do grupo é capaz de explicar
o funcionamento de cada trecho na arguição.

Ferramentas utilizadas
FerramentaDesenvolvedorEtapas de usoClaude (Sonnet)AnthropicEstruturação do projeto, divisão de tarefas, geração do notebook de benchmark, redação das seções do relatório SBC (Metodologia, Resultados, Conclusões), arquivo USO_DE_IA.mdNotebookLMGoogleGeração dos dois decks de slides a partir do relatório e dos notebooks do projetoChatGPTOpenAIEsclarecimento de conceitos CUDA, revisão de código, geração de exemplos e testesGoogle GeminiGoogleApoio na documentação em Markdown e sugestões de boas práticas

Exemplos representativos de prompts e análise crítica
Prompt 1 — Claude (estruturação do projeto)

"Crie um guia para realizar o Trabalho Final — Aceleração de Aplicações
com GPU (CUDA) [...] dividido em fases com tarefas por integrante."

O que foi aproveitado: O guia interativo com as 7 fases e a divisão de
responsabilidades por membro foi utilizado diretamente como referência de
planejamento do grupo.
O que foi ajustado: A distribuição de pontos da rubrica sugerida pela IA
foi revisada pelo grupo, que redistribuiu algumas tarefas de acordo com a
disponibilidade real de cada integrante.
O que estava incorreto: A IA inicialmente não considerou que o Blelloch
e o Hillis-Steele nas implementações eram single-block, o que limitou os
tamanhos de entrada no benchmark. Isso foi identificado e corrigido manualmente
antes de rodar os experimentos.

Prompt 2 — Claude (geração do notebook de benchmark)

"Gere o notebook completo da Fase 3 com medição via cudaEvent, loop de
repetições, cálculo de média/desvio padrão, speedup e exportação de CSV
para resultados/"

O que foi aproveitado: A estrutura completa do notebook (16 células) foi
utilizada, incluindo a célula de validação de corretude (Célula 8), que foi
considerada essencial para garantir que nenhuma kernel incorreta fosse
benchmarkada.
O que foi ajustado: Os tamanhos de entrada para os algoritmos de scan
foram limitados a N=512 (tamanho do bloco) após identificar que as kernels
single-block não suportavam entradas maiores. Essa limitação não estava
prevista no notebook gerado inicialmente.
O que estava incorreto: A função reducao_ingenua gerada precisou ser
adaptada para suportar múltiplos blocos corretamente, pois a versão inicial
assumia um único bloco para qualquer tamanho de entrada.

Prompt 3 — NotebookLM (geração dos slides)

"Com base nas fontes adicionadas, crie uma apresentação de 20 slides para
defesa acadêmica com este roteiro: Slide 1 capa [...] Slides 13-15
resultados com os gráficos reais do benchmark..."

O que foi aproveitado: Dois decks completos foram gerados com estilos
visuais distintos. O deck escuro (20 slides) foi escolhido para a defesa
por conter os dados experimentais reais e seguir a estrutura acadêmica
exigida pela rubrica.
O que foi descartado: O deck claro (14 slides) foi descartado como
apresentação principal por não conter os resultados experimentais, mas
3 slides visuais foram aproveitados e integrados ao deck principal
(diagramas de Hillis-Steele, Blelloch e árvore de decisão).
O que estava incorreto: O NotebookLM não incluiu automaticamente as
fórmulas da Lei de Amdahl e de Gustafson nos slides de fundamentação.
Esses conteúdos foram inseridos manualmente pelo grupo após revisão.

Prompt 4 — ChatGPT (conceitos CUDA)

"Uma dúvida, durante a apresentação, como eu compilaria este código no
notebook Colab online?"

O que foi aproveitado: A orientação sobre como ativar a GPU no Colab
(Runtime → Change runtime type → GPU) e rodar as células em ordem foi
correta e utilizada diretamente.
O que foi ajustado: A resposta não mencionava o risco de desconexão do
Colab em sessões longas. O grupo adicionou a prática de salvar os resultados
na pasta resultados/ e baixá-los antes de encerrar a sessão.

Reflexão individual
Ruan Dener Souza Silva:
O uso da IA, especialmente o Claude, acelerou significativamente a parte de
benchmark — o notebook da Fase 3 foi gerado em minutos e só precisei
adaptar os tamanhos de entrada e validar a corretude das kernels. O que
aprendi de mais importante foi que a IA gera código funcional mas não
necessariamente correto para o contexto específico do problema: a limitação
do single-block precisou ser identificada e corrigida por mim. Isso reforçou
que entender o código é insubstituível.
Gustavo Santos Oliveira:
Utilizei o ChatGPT principalmente para esclarecer dúvidas sobre divergência
de warp e o funcionamento interno do __syncthreads(). As respostas foram
um bom ponto de partida, mas precisei consultar a documentação oficial da
NVIDIA para confirmar detalhes sobre a arquitetura T4 e os limites de
shared memory por bloco. A IA não substitui a documentação primária.
Olívier Queirós Pereira:
O Claude foi útil para estruturar as seções do relatório SBC, especialmente
a Fundamentação Teórica. No entanto, precisei reescrever boa parte do texto
para adequar ao estilo acadêmico e verificar cada referência bibliográfica
manualmente — duas referências sugeridas inicialmente pela IA não existiam
nas fontes indicadas e foram removidas. Isso demonstra que a IA não pode
ser usada como fonte bibliográfica primária.