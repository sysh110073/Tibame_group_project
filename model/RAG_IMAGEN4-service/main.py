import os
import base64
import uuid
import time
import requests
import json
from flask import Flask, request, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from pinecone import Pinecone

app = Flask(__name__, static_folder='static')
if not os.path.exists('static'): os.makedirs('static')

# 環境變數由 Cloud Run 注入
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

qa_chain = None
embeddings_model = None
pc_client = None
index_name = "recipe-vector"

# 簡單的 In-Memory Cache (用於暫存生成的食譜，讓 View Recipe 按鈕可以抓到)
# 結構: { "uuid": { "text": "...", "image_url": "...", "timestamp": 1234567890 } }
generated_recipes = {}

def init_rag_system():
    global qa_chain, embeddings_model, pc_client
    try:
        print("🚀 初始化 Google RAG 系統 (768維度)...")
        
        # 0. 初始化 Pinecone Client (原生)
        if PINECONE_API_KEY:
            pc_client = Pinecone(api_key=PINECONE_API_KEY)
            
        # 1. 使用 Google Embedding 模型
        embeddings_model = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
        
        # 2. 連接 Pinecone (LangChain 用)
        vector_store = PineconeVectorStore.from_existing_index(
            index_name=index_name, 
            embedding=embeddings_model,
            namespace="recipe"
        )
        
        # 3. 建立 LLM
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
        
        # 4. 建立 RAG 鏈
        template = """你是零剩食主廚，請根據資料庫回答：{context} 
        問題：{question}
        回答："""
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            combine_docs_chain_kwargs={"prompt": prompt}
        )
        print("✅ RAG 系統已就緒")
    except Exception as e:
        print(f"❌ 初始化失敗: {e}")

def enrich_recipe_content(recipe_text):
    """
    使用 LLM 萃取/生成:
    1. 好的標題 (Title)
    2. 營養標示 (Nutrition Summary)
    3. 生圖 Prompt (Image Prompt)
    """
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.8)
        
        # 強制要求 JSON 格式
        prompt_template = """
        You are a Michelin Star Chef and Food Stylist.
        Analyze the following recipe or food description:
        "{recipe}"
        
        Please generate a JSON object with the following fields:
        1. "title": A creative and appetizing name for this dish (in Traditional Chinese).
        2. "nutrition": A brief nutritional summary estimate (e.g., "熱量: 500kcal | 蛋白質: 20g") (in Traditional Chinese).
        3. "image_prompt": A high-quality English prompt for Google Imagen 4 (photorealistic, 8k, gourmet style).
        
        Return ONLY the JSON string. Do not include markdown code blocks.
        """
        
        response_text = llm.invoke(prompt_template.format(recipe=recipe_text[:1500])).content.strip()
        
        # 清理可能的回傳格式 (e.g. ```json ... ```)
        if response_text.startswith("```"):
            response_text = response_text.replace("```json", "").replace("```", "")
        
        data = json.loads(response_text)
        return data
        
    except Exception as e:
        print(f"❌ Enrichment Failed: {e}")
        # Fallback values
        return {
            "title": "特製料理",
            "nutrition": "營養滿分",
            "image_prompt": "delicious food, photorealistic, 8k"
        }

def generate_image_prompt(recipe_text):
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.8)
        prompt = """
        You are an expert food photographer. 
        Extract the visual description from the recipe and create a prompt for Google Imagen 4.
        Recipe: {recipe}
        Requirements:
        1. Output ONLY the English prompt.
        2. Include tags: photorealistic, 8k, gourmet food photography, studio lighting, delicious, close-up.
        """
        return llm.invoke(prompt.format(recipe=recipe_text[:1000])).content.strip()
    except Exception as e:
        print(f"❌ Prompt 生成失敗: {e}")
        return "Delicious food, 4k, photorealistic"

# ENV
IMGUR_CLIENT_ID = os.environ.get('IMGUR_CLIENT_ID', '546c25a59c58ad7') # Public demo key

def upload_to_imgur(image_data):
    """Upload binary image data to Imgur and return URL"""
    url = "https://api.imgur.com/3/image"
    headers = {"Authorization": f"Client-ID {IMGUR_CLIENT_ID}"}
    try:
        # [FIX] base64.b64encode returns bytes, Imgur needs string
        b64_string = base64.b64encode(image_data).decode('utf-8')
        response = requests.post(url, headers=headers, data={"image": b64_string}, timeout=30)
        if response.status_code == 200:
            return response.json()['data']['link']
        else:
            print(f"⚠️ Imgur Upload Failed ({response.status_code}): {response.text[:200]}")
    except Exception as e:
        print(f"❌ Imgur Error: {e}")
    return None

def call_imagen_api(prompt):
    print(f"🎨 呼叫 Google Imagen 4 (Fast): {prompt[:30]}...")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/imagen-4.0-fast-generate-001:predict?key={GOOGLE_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "instances": [{"prompt": prompt}],
        "parameters": {"sampleCount": 1, "aspectRatio": "1:1"}
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            b64_image = result.get('predictions', [{}])[0].get('bytesBase64Encoded')
            if b64_image:
                img_data = base64.b64decode(b64_image)
                
                # [FIX] Upload to Imgur instead of local save
                imgur_url = upload_to_imgur(img_data)
                if imgur_url:
                    print(f"✅ 圖片已上傳 Imgur: {imgur_url}")
                    return imgur_url
                
                # Fallback to local if Imgur fails (still ephemeral but better than nothing)
                filename = f"{uuid.uuid4()}.png"
                file_path = os.path.join('static', filename)
                with open(file_path, 'wb') as f:
                    f.write(img_data)
                base_url = request.host_url.replace('http://', 'https://')
                return f"{base_url}static/{filename}"
                
        else:
            print(f"❌ Google Imagen 失敗 ({response.status_code}): {response.text}")
    except Exception as e:
        print(f"❌ Imagen API Error: {e}")
    return None

# ==========================================
# API Endpoints
# ==========================================

def parse_metadata_text(meta_text):
    """
    從 Pinecone 的 metadata['text'] 解析出結構化資料
    格式假設: "dishname: ...\n網址: ...\n材料: ...|... \n步驟: ... "
    """
    if not meta_text: return {}
    
    data = {"title": "未知食譜", "summary": "", "text": meta_text}
    
    try:
        lines = meta_text.split('\n')
        for line in lines:
            if line.strip().startswith("dishname:"):
                data["title"] = line.split(":", 1)[1].strip()
            elif line.strip().startswith("材料:"):
                data["summary"] = line.split(":", 1)[1].strip()[:100] # 暫用材料當摘要
            elif line.strip().startswith("步驟:"):
                pass 
                
        # 如果沒抓到標題，嘗試用第一行
        if data["title"] == "未知食譜" and lines:
            if ":" in lines[0]:
                data["title"] = lines[0].split(":", 1)[1].strip()
            else:
                data["title"] = lines[0].strip()
                
    except Exception as e:
        print(f"⚠️ Metadata Parse Error: {e}")
        
    return data

@app.route('/api/chef', methods=['POST'])
def chef_api():
    """RAG Generation + Retrieval for Carousel (ALL with Images & Nutrition)"""
    try:
        if qa_chain is None: 
            init_rag_system()
        if qa_chain is None:
            return jsonify({"error": "RAG System not initialized", "recipes": []}), 500
        
        data = request.json
        user_input = data.get('message', '')
        ingredients_only = data.get('ingredients') # [New] Explicit ingredients
        
        from concurrent.futures import ThreadPoolExecutor
        
        # 1. Generate Main Recipe (AI Chef)
        # ---------------------------------
        res = qa_chain.invoke({"question": user_input, "chat_history": []})
        gen_text = res["answer"]
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Task A: Enrich Main Recipe
            future_enrich_main = executor.submit(lambda: enrich_recipe_content(gen_text))
            
            # 2. Retrieve Similar
            recipes = []
            gen_id = str(uuid.uuid4())
            
            # [FIX] Use explicit ingredients for vector search if available
            query_text = ingredients_only if ingredients_only else user_input
            print(f"🔍 Searching recipes with query: {query_text}")
            gen_vector = embeddings_model.embed_query(query_text) 
            
            # [IMPROVED] Parse user ingredients - split by comma OR slash
            import re
            user_ingredients_list = [ing.strip().lower() for ing in re.split(r'[,/]', query_text) if ing.strip()]
            print(f"📋 User ingredients for filtering: {user_ingredients_list}")
            
            query_res = None
            if pc_client:
                try:
                    idx = pc_client.Index(index_name)
                    # Fetch MORE candidates for post-filtering
                    query_res = idx.query(
                        vector=gen_vector,
                        top_k=15, # Larger pool for filtering
                        include_metadata=True,
                        namespace="recipe" 
                    )
                except Exception as e:
                    print(f"⚠️ Retrieval failed: {e}")

            # Helper function to check ingredient containment
            def recipe_contains_ingredients(recipe_text, ingredients_list):
                """Check if recipe_text contains at least one of the user's ingredients"""
                recipe_lower = recipe_text.lower()
                for ing in ingredients_list:
                    if ing in recipe_lower:
                        return True
                return False
                    
            # Task B & C: Enrich Retrieved Recipes (with strict filtering)
            retrieved_futures = []
            matched_count = 0
            if query_res and query_res.matches:
                print(f"📡 Pinecone found {len(query_res.matches)} matches")
                for match in query_res.matches:
                    if matched_count >= 4: # Increased to 4 DB recipes
                        break
                        
                    meta = match.metadata or {}
                    raw_text = meta.get('text', '')
                    
                    # [STRICT FILTER] Only include if recipe contains user's ingredients
                    if user_ingredients_list and not recipe_contains_ingredients(raw_text, user_ingredients_list):
                        print(f"⏭️ Skipping recipe (no match for {user_ingredients_list}): {raw_text[:50]}...")
                        continue
                        
                    print(f"✅ Recipe matched: {raw_text[:50]}... (Score: {match.score})")
                    matched_count += 1
                    
                    parsed = parse_metadata_text(raw_text)
                    combined_info = f"{parsed['title']}\n{raw_text[:500]}"
                    
                    future_enrich = executor.submit(lambda txt=combined_info: enrich_recipe_content(txt))
                    retrieved_futures.append({"match": match, "future": future_enrich})
            else:
                print("⚠️ No recipes found in Pinecone for this query.")

            # ---------------------------------------------------------
            # [OPTIMIZATION] Parallelize Image Generation for ALL recipes
            # ---------------------------------------------------------
            
            # 1. Get Enrichment Results (Blocking slightly, but effectively parallel LLM calls)
            enriched_main = future_enrich_main.result()
            enriched_retrieved = []
            for item in retrieved_futures:
                enriched_retrieved.append({
                    "match": item['match'],
                    "data": item['future'].result()
                })
                
            # 2. Submit Image Generation Tasks in Parallel
            # We use a NEW ThreadPool to ensure image calls (IO bound) don't block each other
            img_futures = []
            
            # Main Recipe Image
            future_img_main = executor.submit(lambda: call_imagen_api(enriched_main['image_prompt']))
            
            # Retrieved Recipes Images
            for item in enriched_retrieved:
                prompt = item['data']['image_prompt']
                fut = executor.submit(lambda p=prompt: call_imagen_api(p))
                img_futures.append({"item": item, "future": fut})
                
            # 3. Collect Results
            main_image_url = future_img_main.result()
            
            # Save Generated Recipe
            if pc_client:
                try:
                    idx = pc_client.Index(index_name)
                    idx.upsert(
                        vectors=[{
                            "id": gen_id, 
                            "values": gen_vector, 
                            "metadata": {
                                "type": "generated",
                                "title": enriched_main['title'],
                                "text": gen_text, 
                                "nutrition": enriched_main['nutrition'],
                                "image_url": main_image_url or "",
                                "created_at": time.time()
                            }
                        }],
                        namespace="generated_cache"
                    )
                except Exception as e: pass

            recipes.append({
                "id": gen_id,
                "title": enriched_main['title'],
                "summary": enriched_main['nutrition'],
                "image_url": main_image_url,
                "source": "generated"
            })
            
            # Collect Retrieved Images
            for img_job in img_futures:
                item = img_job['item']
                img_url = img_job['future'].result()
                match = item['match']
                enriched_data = item['data']
                
                recipes.append({
                    "id": match.id,
                    "title": enriched_data['title'],
                    "summary": enriched_data['nutrition'], 
                    "image_url": img_url,
                    "source": "db"
                })

        return jsonify({"recipes": recipes})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "recipes": []}), 500

@app.route('/api/recipe', methods=['GET'])
def get_recipe():
    """Fetch full recipe details (Query Param: ?id=...)"""
    recipe_id = request.args.get('id')
    if not recipe_id: return jsonify({"error": "Missing id"}), 400
    
    # [Fix] URL decoding might turn + into space. Restore it.
    if ' ' in recipe_id:
        recipe_id = recipe_id.replace(' ', '+')
        
    print(f"🔍 Fetching Recipe ID: {recipe_id}")

    try:
        if pc_client:
            idx = pc_client.Index(index_name)
            
            # 1. Try generated_cache
            fetch_res = idx.fetch(ids=[recipe_id], namespace="generated_cache")
            if recipe_id in fetch_res.vectors:
                data = fetch_res.vectors[recipe_id].metadata
                return jsonify({
                    "title": data.get("title", "Recipe"),
                    "text": data.get("text"),
                    "image_url": data.get("image_url")
                })
            
            # 2. Try recipe namespace (DB)
            fetch_res_db = idx.fetch(ids=[recipe_id], namespace="recipe")
            if recipe_id in fetch_res_db.vectors:
                print("✅ Found in DB by ID")
                data = fetch_res_db.vectors[recipe_id].metadata
                raw_text = data.get("text", "")
                parsed = parse_metadata_text(raw_text)
                return jsonify({
                    "title": parsed.get("title", "美味食譜"),
                    "text": raw_text, 
                    "image_url": "https://via.placeholder.com/300?text=Delicious"
                })
            
            # 3. Fallback: If ID fetch fails, try to decode ID (since it's B64 Title) and Query
            print("⚠️ ID fetch failed, attempting Query Fallback...")
            try:
                # Try to decode Base64 ID to get Dish Name
                decoded_title = base64.b64decode(recipe_id).decode('utf-8')
                print(f"🔄 Decoded Title: {decoded_title}")
                
                # Embedding search for this title
                vec = embeddings_model.embed_query(decoded_title)
                query_res = idx.query(
                    vector=vec, 
                    top_k=1, 
                    namespace="recipe", 
                    include_metadata=True
                )
                if query_res.matches:
                    match = query_res.matches[0]
                    # Score check? maybe > 0.9 for exact match
                    if match.score > 0.85:
                        print(f"✅ Found via Query Fallback (Score: {match.score})")
                        data = match.metadata
                        raw_text = data.get("text", "")
                        parsed = parse_metadata_text(raw_text)
                        return jsonify({
                            "title": parsed.get("title", "美味食譜"),
                            "text": raw_text, 
                            "image_url": "https://via.placeholder.com/300?text=Delicious"
                        })
            except Exception as e_fallback:
                print(f"❌ Fallback failed: {e_fallback}")

    except Exception as e:
        print(f"❌ Get recipe error: {e}")

    return jsonify({"error": "Recipe not found"}), 404


@app.route('/api/store_recipe', methods=['POST'])
def store_recipe():
    """將生成的食譜存入 Pinecone 的 generated_cache 命名空間"""
    data = request.get_json()
    recipe_id = data.get('recipe_id')
    text = data.get('text')
    title = data.get('title', 'Generated Recipe')
    
    if not recipe_id or not text:
        return jsonify({"error": "Missing recipe_id or text"}), 400
        
    try:
        if pc_client:
            idx = pc_client.Index(index_name)
            # 生成 Embedding
            vector = embeddings_model.embed_query(text[:1500])
            
            idx.upsert(
                vectors=[{
                    "id": recipe_id,
                    "values": vector,
                    "metadata": {
                        "type": "generated",
                        "title": title,
                        "text": text,
                        "created_at": time.time()
                    }
                }],
                namespace="generated_cache"
            )
            return jsonify({"status": "success", "recipe_id": recipe_id})
    except Exception as e:
        print(f"❌ Store recipe error: {e}")
        return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": "Pinecone not initialized"}), 500


@app.route('/api/like', methods=['POST'])
def like_recipe():
    """Positive Feedback: Update User Vector"""
    if pc_client is None: init_rag_system()
    
    data = request.json
    user_id = data.get('user_id')
    recipe_id = data.get('recipe_id')
    
    if not user_id or not recipe_id: return jsonify({"error": "Missing params"}), 400
    
    print(f"❤️ User {user_id} liked {recipe_id}")
    
    try:
        idx = pc_client.Index(index_name)
        recipe_vector = None
        
        # 1. Get Recipe Vector (Check gen, then db)
        fetch_gen = idx.fetch(ids=[recipe_id], namespace="generated_cache")
        if recipe_id in fetch_gen.vectors:
            recipe_vector = fetch_gen.vectors[recipe_id].values
        else:
            fetch_db = idx.fetch(ids=[recipe_id], namespace="recipe")
            if recipe_id in fetch_db.vectors:
                 recipe_vector = fetch_db.vectors[recipe_id].values

        if not recipe_vector:
             return jsonify({"status": "skipped_no_vector"}), 200

        # 2. Update User Vector (namespace='users')
        fetch_user = idx.fetch(ids=[user_id], namespace="users")
        
        if user_id in fetch_user.vectors:
            old_vector = fetch_user.vectors[user_id].values
            # Logic: New = Old * 0.8 + Recipe * 0.2
            new_vector = [o * 0.8 + n * 0.2 for o, n in zip(old_vector, recipe_vector)]
        else:
            # First time user
            new_vector = recipe_vector
            
        idx.upsert(
            vectors=[(user_id, new_vector)],
            namespace="users"
        )
        return jsonify({"status": "updated"})
        
    except Exception as e:
        print(f"❌ Like failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/random_recommend', methods=['POST'])
def random_recommend():
    """Personalized Recommendation (User Vector -> DB Search)"""
    if pc_client is None: init_rag_system()
    
    data = request.json
    user_id = data.get('user_id')
    ingredients = data.get('ingredients') # [New] Context
    
    try:
        idx = pc_client.Index(index_name)
        
        # 1. Determine Query Vector
        if ingredients:
            print(f"🥦 Contextual Recommend for {user_id} with ingredients: {ingredients}")
            query_vector = embeddings_model.embed_query(ingredients)
        else:
            # 1b. User History or Cold Start
            fetch_res = idx.fetch(ids=[user_id], namespace="users")
            if user_id in fetch_res.vectors:
                query_vector = fetch_res.vectors[user_id].values
                print(f"👤 Personalized for {user_id}")
            else:
                import random
                topics = ["Taiwanese Cuisine", "Healthy Salad", "Pasta", "Japanese Food", "Dessert", "Seafood"]
                topic = random.choice(topics)
                print(f"👤 Cold Start for {user_id}, topic: {topic}")
                query_vector = embeddings_model.embed_query(topic)
        
        # 2. Search DB with larger K for variation
        # [FIX] Fetch large pool for post-filtering
        
        query_res = idx.query(
            vector=query_vector,
            top_k=30, # Larger pool for filtering
            include_metadata=True,
            namespace="recipe" 
        )
        
        # Parse user ingredients for strict filtering
        user_ingredients_list = []
        if ingredients:
            user_ingredients_list = [ing.strip().lower() for ing in ingredients.split(',') if ing.strip()]
            print(f"📋 Filtering by ingredients: {user_ingredients_list}")
        
        # Helper function to check ingredient containment
        def recipe_contains_ingredients(recipe_text, ingredients_list):
            """Check if recipe_text contains at least one of the user's ingredients"""
            recipe_lower = recipe_text.lower()
            for ing in ingredients_list:
                if ing in recipe_lower:
                    return True
            return False
        
        # 3. Filter and Process Matches
        matches = query_res.matches
        filtered_matches = []
        
        for match in matches:
            if len(filtered_matches) >= 3:
                break
            meta = match.metadata or {}
            raw_text = meta.get('text', '')
            
            # If we have ingredients, apply strict filter
            if user_ingredients_list:
                if recipe_contains_ingredients(raw_text, user_ingredients_list):
                    print(f"✅ Match (ingredient): {raw_text[:50]}...")
                    filtered_matches.append(match)
                else:
                    print(f"⏭️ Skip (no match): {raw_text[:50]}...")
            else:
                # No ingredients = cold start, accept any
                filtered_matches.append(match)
        
        matches = filtered_matches
        
        # [FIX] Use ThreadPool to Enrich Content (Title/Nutrition/Image)
        # This fixes the "blank" issue by properly parsing and generating missing info
        from concurrent.futures import ThreadPoolExecutor
        recommendations = []
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for match in matches:
                meta = match.metadata or {}
                raw_text = meta.get('text', '')
                parsed = parse_metadata_text(raw_text)
                
                # Enrich context
                combined_info = f"{parsed['title']}\n{raw_text[:500]}"
                future = executor.submit(lambda txt=combined_info: enrich_recipe_content(txt))
                
                futures.append({
                    "match": match, 
                    "parsed": parsed, 
                    "future": future
                })
            
            for item in futures:
                match = item['match']
                enriched = item['future'].result()
                
                # We can skip image gen if it's too slow, but for quality let's do it or use placeholder
                # User asked for high quality output. Let's try to gen image if fast enough, 
                # or maybe just reuse a generic one if timeout.
                # Here we call imagen (might take 2-3s per image)
                img_url = call_imagen_api(enriched['image_prompt'])
                
                recommendations.append({
                    "id": match.id,
                    "title": enriched['title'],
                    "summary": enriched['nutrition'],
                    "image_url": img_url,
                    "source": "db"
                })
            
        return jsonify({"recipes": recommendations})
        
    except Exception as e:
        print(f"❌ Recommend failed: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    init_rag_system()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))