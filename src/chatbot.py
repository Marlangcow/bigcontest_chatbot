from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from typing import List
from langchain.schema import Document


def get_chatbot_response(user_input, memory, chain, retrievers):
    search_results = flexible_function_call_search(user_input, retrievers)
    search_results_str = "\n".join([doc.page_content for doc in search_results]).strip()

    if not search_results_str:
        return "검색된 내용이 없어서 답변을 드릴 수 없습니다."

    chat_history = memory.load_memory_variables({}).get("chat_history", "")

    input_data = {
        "input_text": user_input,
        "search_results": search_results_str,
        "chat_history": chat_history,
    }

    try:
        output = chain(input_data)
        output_text = output.get("text", str(output))
    except Exception as e:
        print(f"LLM 응답 생성 중 오류 발생: {e}")
        return "응답을 생성하는 과정에서 오류가 발생했습니다. 다시 시도해주세요."

    memory.save_context({"input": user_input}, {"output": output_text})
    return output_text


def flexible_function_call_search(user_input: str, retrievers: List) -> List[Document]:
    """
    사용자 입력을 기반으로 여러 retriever를 사용하여 관련 문서를 검색합니다.

    Args:
        user_input (str): 사용자 입력 텍스트
        retrievers (List): 사용할 retriever들의 리스트

    Returns:
        List[Document]: 검색된 문서들의 리스트
    """
    all_results = []

    for retriever in retrievers:
        try:
            results = retriever.get_relevant_documents(user_input)
            all_results.extend(results)
        except Exception as e:
            print(f"Retriever 검색 중 오류 발생: {e}")
            continue

    return all_results
