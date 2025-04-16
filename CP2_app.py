import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pycaret.classification import load_model, predict_model

st.set_page_config( page_title = 'Simulador - Case Ifood',
                    page_icon = './images/logo_fiap.png',
                    layout = 'wide',
                    initial_sidebar_state = 'expanded')


model = load_model('./Model Training/saved_models/37')
gen_ai_path = os.getenv('QWEN2_5_3B_INSTRUCT') # O path do modelo tal como nas variáveis de ambiente da máquina do autor.

with open("ptbr_app_description.txt", "r", encoding="utf-8") as ptbr_app_description:
    descricao = ptbr_app_description.read()

with st.expander('Descrição do App', expanded=False):
    st.write(descricao)

with st.sidebar:
    c1, c2 = st.columns([.3, .7])
    c1.image('./images/logo_fiap.png', width = 100)
    c2.write('')
    c2.subheader('Auto ML - Fiap [v1]')
    database = st.radio('Fonte dos dados de entrada:', ('CSV', 'Online'), horizontal = True)
    

    st.write(" ")
    st.write(" ")

    threshold = st.session_state.threshold_inferido = 0.5

    if gen_ai_path is not None:
        st.write("✨ Gerar threshold via Generative AI ✨")
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import torch

        # Cache do modelo carregado
        @st.cache_resource
        def carregar_modelo():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            tokenizer = AutoTokenizer.from_pretrained(gen_ai_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                gen_ai_path,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            return pipeline("text-generation", model=model, tokenizer=tokenizer)

        pipe = carregar_modelo()

        @st.cache_data(show_spinner="Analisando intenção com IA...")
        def gerar_threshold(frase_usuario: str, prompt_base: str):
            prompt_final = prompt_base.format(frase=frase_usuario)
            resposta = pipe(prompt_final, max_new_tokens=10, do_sample=False)[0]['generated_text']
            resposta = resposta.replace(prompt_final, '').strip()

            try:
                resposta_filtrada = ''.join(c for c in resposta if c in '0123456789.')
                valor = float(resposta_filtrada)
                if 0 <= valor <= 1:
                    return valor
            except:
                pass
            return 0.5  # fallback padrão

        @st.cache_data(show_spinner="Gerando explicação personalizada...")
        def gerar_explicacao(frase_usuario: str, threshold: float, prompt_explicacao: str):
            prompt = prompt_explicacao.format(frase=frase_usuario, threshold=threshold)
            explicacao_raw = pipe(
                prompt,
                max_new_tokens=150,
                temperature=0.3,
                repetition_penalty=1.2,
                do_sample=False
            )[0]['generated_text']
            
            explicacao = explicacao_raw.split('[INÍCIO DA EXPLICAÇÃO]')[-1].strip()
            frases = explicacao.split('. ')
            # o número 3 abaixo serve para cortar a geração e evitar prolixidade.
            return '. '.join(frases[:3]) + '.' if len(frases) > 1 else explicacao

        # Leitura dos arquivos de prompt
        with open("ptbr_threshold_prompt.txt", "r", encoding="utf-8") as ptbr_threshold_prompt:
            PROMPT_BASE = ptbr_threshold_prompt.read()

        with open("ptbr_explanation_prompt.txt", "r", encoding="utf-8") as ptbr_explanation_prompt:
            PROMPT_EXPLICACAO = ptbr_explanation_prompt.read()

        # Entrada do usuário
        frase = st.text_input(
            "Diga com suas palavras o quão conservador ou agressivo deseja ser na predição de 'comprar'. Você também pode pedir por um valor específico entre 0 e 1:",
            placeholder="Ex: Prefiro evitar falsos positivos"
        )

        st.session_state.threshold_valido = False

        if frase:
            valor = gerar_threshold(frase, PROMPT_BASE)
            st.session_state.threshold_inferido = valor
            st.session_state.threshold_valido = True if valor != 0.5 or frase.strip() == "0.5" else False
            st.success(f"Threshold sugerido pela IA: **{valor:.2f}**")
        else:
            st.warning("Threshold padrão: 0.5")

        if frase and st.session_state.threshold_valido:
            threshold = st.session_state.threshold_inferido
            explicacao = gerar_explicacao(frase, threshold, PROMPT_EXPLICACAO)

            with st.expander("📘 Por que este threshold foi sugerido para você?"):
                st.markdown(explicacao)
    else:
        st.write("A opção de threshold via generative AI não está disponível em produção. 😕")
        threshold = st.slider(
            'Threshold (ponto de corte para considerar predição como "comprar")',
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            value=st.session_state.threshold_inferido,
            key="threshold_slider"
        )

if database == 'CSV':
    st.title('Simulador de Conversão de Vendas com CSV')
elif database == 'Online':
    st.title('Simulador de Conversão de Vendas com Dados Online')

# OneHotEncoding que foi feito fora do pycaret para tratar os dados categóricos
def aplicar_onehot_xtest(Xtest):
    object_cols = Xtest.select_dtypes(include='object').columns
    with open('Model Training/one-hot-encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    dados_transformados = encoder.transform(Xtest[object_cols])
    df_test_encoded = pd.DataFrame(dados_transformados, columns=encoder.get_feature_names_out(object_cols), index=Xtest.index)
    Xtest = pd.concat([Xtest.drop(object_cols, axis=1), df_test_encoded], axis=1)
    return Xtest

if database == 'CSV':
    st.info('Upload do CSV')
    arquivo_csv = st.file_uploader('Selecione o arquivo CSV', type='csv')
       

    if arquivo_csv:
        Xtest_raw = pd.read_csv(arquivo_csv)
        Xtest = aplicar_onehot_xtest(Xtest_raw.copy())
        ypred = predict_model(model, data=Xtest, raw_score=True)

        tabs = st.tabs(['Visualizar Predições', 'Analytics'])

        with tabs[0]:
            c1, c2, c3, c4 = st.columns(4)
            qtd_true = ypred.loc[ypred['prediction_score_1'] > threshold].shape[0]
            c2.metric('Qtd clientes True', value=qtd_true)
            c3.metric('Qtd clientes False', value=len(ypred) - qtd_true)

            def color_pred(val):
                color = 'olive' if val > threshold else 'orangered'
                return f'background-color: {color}'

            tipo_view = st.radio('', ('Completo', 'Apenas predições'))
            df_view = ypred.copy() if tipo_view == 'Completo' else pd.DataFrame(ypred.iloc[:, -1].copy())

            st.dataframe(df_view.style.applymap(color_pred, subset=['prediction_score_1']))

            csv = df_view.to_csv(sep=';', decimal=',', index=True)
            st.markdown(f'Shape do CSV a ser baixado: {df_view.shape}')
            st.download_button(label='Download CSV',
                               data=csv,
                               file_name='Predicoes.csv',
                               mime='text/csv')

        with tabs[1]: 
            st.subheader("Análise Comparativa das Features")
            ypred_copy = ypred.copy()
            ypred_copy['prediction_label'] = (ypred_copy['prediction_score_1'] > threshold).astype(int)

            try:
                Xtest_raw.reset_index(drop=True, inplace=True)
                ypred_copy.reset_index(drop=True, inplace=True)
                df_analytics = pd.concat([Xtest_raw, ypred_copy['prediction_label']], axis=1)
            except Exception as e:
                st.error(f"Erro ao montar dataframe para análise: {e}")

            plot_type = st.radio("Escolha o tipo de gráfico:", options=["Boxplot", "Histograma"])

            numeric_features = df_analytics.select_dtypes(include=['int64', 'float64']).drop(columns=['Response'], errors='ignore').columns

            n = len(numeric_features)
            cols = 3  # quantidade de colunas no layout
            rows = (n + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(10 * cols, 5 * rows))
            axes = axes.flatten()

            for i, col in enumerate(numeric_features):
                if plot_type == "Boxplot":
                    sns.boxplot(x='prediction_label', y=col, data=df_analytics, ax=axes[i])
                    axes[i].set_xlabel("Predição")
                    axes[i].set_ylabel(col)
                else:  # Histograma
                    for label in sorted(df_analytics['prediction_label'].unique()):
                        subset = df_analytics[df_analytics['prediction_label'] == label]
                        axes[i].hist(subset[col], bins=30, alpha=0.5, label=f'Predição {label}')
                    axes[i].legend()
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel("Frequência")

                axes[i].set_title(f'Distribuição da feature "{col}"')

            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            st.pyplot(fig)

    else:
        st.warning('Arquivo CSV não foi carregado')
        with st.expander('Conteúdo esperado do Arquivo CSV', expanded=False):
            catalog = pd.read_csv('catalog.csv')
            st.dataframe(catalog)
            exemplo=pd.read_csv('Model Training/cleansed_df_test.csv')
            exemplo.drop(columns=['Response'], inplace=True)
            c1, c2 = st.columns([.5, .5])
            with c1:
                st.write('Você pode testar com o dataset de exemplo:')
            with c2:
                st.download_button(label = 'Baixar dataset de exemplo',
                            data = exemplo.to_csv(index=False),
                            file_name = 'dados_teste.csv',
                            mime = 'text/csv')
        
if database == 'Online':
    input_data = {}

    st.subheader("Preencha os campos abaixo para a simulação:")
    st.write(" ")

    st.markdown("<h3 style='text-align: center;'>🧍 Informações Pessoais e Relacionamento com a Marca</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        input_data["Age"] = st.slider("Idade", 0, 120, 55)
    with col2:
        input_data["Education"] = st.selectbox(
            "Escolaridade",
            ['2n Cycle', 'Graduation', 'PhD', 'Master', 'Basic'],
            index=1
        )
    with col3:
        input_data["Marital_Status"] = st.selectbox(
            "Estado civil",
            ['Married', 'Together', 'Single', 'Divorced', 'Widow'],
            index=0
        )
    col1, col2, col3 = st.columns(3)
    with col1:
        input_data["Kidhome"] = st.slider("Nº de crianças pequenas", 0, 6, 0)
    with col2:
        input_data["Teenhome"] = st.slider("Nº de adolescentes", 0, 6, 0)
    with col3:
        input_data["Time_Customer"] = st.number_input("Tempo como cliente (dias)", min_value=0, max_value=5000, value=1000)

    input_data["Income"] = st.number_input("Renda anual (em reais)", min_value=0, max_value=2_000_000, value=51563)

    col1, col2 = st.columns(2)

    with col1:
        campanhas = {
        "AcceptedCmp1": "Campanha 1",
        "AcceptedCmp2": "Campanha 2",
        "AcceptedCmp3": "Campanha 3",
        "AcceptedCmp4": "Campanha 4",
        "AcceptedCmp5": "Campanha 5"
        }
        campanhas_aceitas = st.multiselect(
            "Campanhas aceitas pelo cliente",
            list(campanhas.values())
        )
        for var, nome in campanhas.items():
            input_data[var] = 1 if nome in campanhas_aceitas else 0

    with col2:
        st.write(' ')
        st.write(' ')
        input_data["Complain"] = 1 if st.checkbox("Cliente reclamou nos últimos 2 anos") else 0
    
    st.markdown("<h3 style='text-align: center;'>🛒 Gastos com Produtos</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        input_data["MntWines"] = st.slider("Vinhos", 0, 5000, 179)
        input_data["MntFruits"] = st.slider("Frutas", 0, 600, 8)
    with col2:
        input_data["MntFishProducts"] = st.slider("Peixes", 0, 800, 12)
        input_data["MntMeatProducts"] = st.slider("Carnes", 0, 6000, 69)
    with col3:
        input_data["MntSweetProducts"] = st.slider("Doces", 0, 800, 8)
        input_data["MntGoldProds"] = st.slider("Produtos de ouro", 0, 1000, 24)
            
    st.markdown("<h3 style='text-align: center;'>🛍️ Comportamento de Compra</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        input_data["NumCatalogPurchases"] = st.slider("Compras por catálogo", 0, 90, 0)
        input_data["NumDealsPurchases"] = st.slider("Compras com desconto", 0, 50, 0)
    with col2:
        input_data["NumStorePurchases"] = st.slider("Compras em loja física", 0, 40, 0)
        input_data["NumWebPurchases"] = st.slider("Compras online", 0, 90, 0)
    with col3:
        input_data["NumWebVisitsMonth"] = st.slider("Visitas ao site no mês", 0, 60, 6)
        input_data["Recency"] = st.slider("Dias desde última compra", 0, 300, 50)

    input_df = pd.DataFrame([input_data])
    input_df = aplicar_onehot_xtest(input_df)
    score = predict_model(model, data=input_df, raw_score=True)['prediction_score_1'].iloc[0]

    if score > threshold:
        prediction = 1
    else:
        prediction = 0
        
    prediction = 1 if score > threshold else 0

    if prediction == 1:
        st.markdown(f"""
            <div style="background-color: #28a745; color: white; padding: 20px; border-radius: 5px; font-size: 18px; font-weight: bold;">
                🎯 Cliente propenso a comprar (Score = {score:.2f})
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style="background-color: #dc3545; color: white; padding: 20px; border-radius: 5px; font-size: 18px; font-weight: bold;">
                🚫 Cliente não propenso a comprar (Score = {score:.2f})
            </div>
        """, unsafe_allow_html=True)
