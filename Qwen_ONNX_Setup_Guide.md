# Qwen1.5-0.5B-Chat ONNX 설정 가이드

이 가이드는 `chkim123/Qwen1.5-0.5B-Chat-ONNX` 모델을 Android 프로젝트에 통합하는 전체 과정을 설명합니다.

## 1. 모델 개요

### 1.1 Qwen1.5-0.5B-Chat 특징
- **모델 크기**: 0.5B 파라미터 (약 1.2GB)
- **최대 시퀀스 길이**: 32,768 토큰
- **어휘집 크기**: 151,936 토큰
- **토크나이저**: GPT-2 스타일 BPE
- **채팅 포맷**: `<|im_start|>`, `<|im_end|>` 태그 기반

### 1.2 특수 토큰
```
<|endoftext|>  : 151643 (EOS, BOS, PAD, UNK 공통)
<|im_start|>   : 151644 (채팅 시작 마커)
<|im_end|>     : 151645 (채팅 종료 마커)
```

## 2. 모델 다운로드 및 설정

### 2.1 Python 환경 준비
```bash
pip install huggingface_hub transformers torch
```

### 2.2 모델 다운로드 스크립트 실행
```bash
python convert_qwen_to_onnx.py
```

이 스크립트는 다음을 수행합니다:
1. `chkim123/Qwen1.5-0.5B-Chat-ONNX`에서 ONNX 모델 다운로드
2. 원본 Qwen 토크나이저 다운로드
3. 어휘집 JSON 파일 생성
4. Android assets용 파일 준비

### 2.3 Android 프로젝트에 파일 복사
다운로드 완료 후:
```
android_assets/qwen_model.onnx → app/src/main/assets/qwen_model.onnx
android_assets/qwen_vocab.json → app/src/main/assets/qwen_vocab.json
```

## 3. 채팅 프롬프트 포맷

### 3.1 Qwen1.5 채팅 템플릿
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Translate 'laptop' to French.<|im_end|>
<|im_start|>assistant
```

### 3.2 응답 파싱
모델 응답에서 다음 패턴으로 텍스트 추출:
- `<|im_start|>assistant` 이후 텍스트
- `<|im_end|>` 또는 `<|endoftext|>` 이전 텍스트

## 4. 코드 변경사항 요약

### 4.1 주요 상수 변경
```kotlin
// 이전 (MobiLlama)
MAX_SEQUENCE_LENGTH = 2048
VOCAB_SIZE = 32000
EOS_TOKEN_ID = 2

// 현재 (Qwen1.5)
MAX_SEQUENCE_LENGTH = 32768
VOCAB_SIZE = 151936  
EOS_TOKEN_ID = 151643
```

### 4.2 토크나이저 변경
```kotlin
// 이전: SentencePiece (▁ 공백)
var processedText = text.replace(" ", "▁")

// 현재: GPT-2 BPE (Ġ 공백)
var processedText = text.replace(" ", "Ġ")
```

### 4.3 프롬프트 포맷 변경
```kotlin
// 이전: MobiLlama 스타일
"""### Human: $question
### Assistant:"""

// 현재: Qwen1.5 스타일
"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
$question<|im_end|>
<|im_start|>assistant
"""
```

## 5. 성능 고려사항

### 5.1 메모리 요구사항
- **모델 크기**: ~1.2GB (FP16)
- **추론 메모리**: 추가 ~500MB
- **권장 RAM**: 4GB 이상

### 5.2 처리 속도
- **CPU 추론**: 토큰당 200-500ms (기기별 차이)
- **최적화**: NNAPI, XNNPACK 활용 가능
- **배치 크기**: 1 (모바일 최적화)

### 5.3 시퀀스 길이 관리
```kotlin
// 실제 사용에서는 더 짧은 길이 권장
private val MOBILE_MAX_LENGTH = 512  // 32768 대신 사용
private val MAX_GENERATION_LENGTH = 20
```

## 6. 문제 해결

### 6.1 일반적인 오류

**OutOfMemoryError:**
- 시퀀스 길이 줄이기 (`MAX_SEQUENCE_LENGTH = 1024`)
- 생성 길이 제한 (`MAX_GENERATION_LENGTH = 10`)
- 앱 힙 크기 증가 (`android:largeHeap="true"`)

**모델 로드 실패:**
```bash
# 파일 크기 확인
ls -lh app/src/main/assets/qwen_model.onnx

# 100MB 초과시 분할 또는 압축 고려
```

**토큰화 오류:**
- 어휘집 파일 크기 확인 (약 600KB)
- JSON 형식 유효성 검사
- 특수 토큰 ID 확인

### 6.2 성능 최적화

**1. ONNX Runtime 설정:**
```kotlin
val sessionOptions = SessionOptions()
sessionOptions.setIntraOpNumThreads(2)  // CPU 코어 수에 맞게 조정
sessionOptions.addConfigEntry("session.inter_op_thread_pool_size", "2")
```

**2. 실행 제공자 최적화:**
```kotlin
// CPU 기본
sessionOptions.addConfigEntry("providers", "CPUExecutionProvider")

// Android NNAPI (지원되는 경우)
sessionOptions.addConfigEntry("providers", "NnapiExecutionProvider,CPUExecutionProvider")
```

**3. 메모리 관리:**
```kotlin
// 명시적 리소스 해제
inputTensor.close()
attentionTensor.close()
outputs.forEach { it.close() }
```

## 7. 대안 접근법

### 7.1 모델이 너무 큰 경우
1. **양자화 버전 사용**:
   - INT8 양자화로 크기 50% 감소
   - 정확도 약간 감소하지만 속도 향상

2. **더 작은 모델 고려**:
   - TinyLlama-1.1B-Chat
   - Qwen1.5-1.8B (더 크지만 성능 향상)

### 7.2 하이브리드 접근
```kotlin
suspend fun generateResponse(query: String): String {
    return if (isComplexQuery(query) && hasNetworkConnection()) {
        // 복잡한 쿼리는 서버 API 사용
        callRemoteAPI(query)
    } else {
        // 간단한 쿼리는 온디바이스 모델 사용
        generateWithQwen(query)
    }
}
```

## 8. 참고 자료

- [Qwen1.5 모델 카드](https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat)
- [ONNX Runtime Android 문서](https://onnxruntime.ai/docs/tutorials/mobile/)
- [Hugging Face ONNX 변환 가이드](https://huggingface.co/docs/optimum/onnxruntime/overview)

## 9. 다음 단계

1. **모델 다운로드**: `python convert_qwen_to_onnx.py` 실행
2. **파일 복사**: assets 폴더에 모델 파일 배치
3. **빌드 테스트**: Android 앱 빌드 및 실행
4. **성능 모니터링**: 메모리 사용량 및 응답 시간 확인
5. **최적화 적용**: 필요시 시퀀스 길이 조정 