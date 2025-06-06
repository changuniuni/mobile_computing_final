# ONNX Runtime Mobile 설정 가이드

이 가이드는 현재 TensorFlow Lite 기반 코드를 ONNX Runtime Mobile로 변경하는 전체 과정을 설명합니다.

## 1. 필요한 변경사항 요약

### 1.1 Gradle 의존성 변경
```gradle
// 기존 TensorFlow Lite 제거
// implementation "org.tensorflow:tensorflow-lite:2.14.0"

// ONNX Runtime Android 추가
implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.17.1'
```

### 1.2 코드 변경사항
- TensorFlow Lite Interpreter → ONNX Runtime OrtSession
- .tflite 파일 → .onnx 파일
- ByteBuffer 입력 → OnnxTensor 입력
- 메모리 관리 자동화 → 명시적 리소스 해제

## 2. MobiLlama 모델 ONNX 변환

### 2.1 Python 환경 설정
```bash
pip install torch transformers optimum[onnxruntime]
```

### 2.2 변환 스크립트 실행
```bash
python convert_mobillama_to_onnx.py
```

이 스크립트는 다음을 수행합니다:
1. MobiLlama-05B-Chat 모델 다운로드
2. ONNX 형식으로 변환
3. 어휘집 JSON 파일 생성
4. Android assets용 파일 준비

## 3. Android 프로젝트 설정

### 3.1 파일 복사
변환 완료 후 다음 파일들을 복사:
```
android_assets/mobillama_model.onnx → app/src/main/assets/
android_assets/mobillama_vocab.json → app/src/main/assets/
```

### 3.2 ProGuard 설정 (선택사항)
`proguard-rules.pro`에 ONNX Runtime 관련 규칙 추가:
```
-keep class ai.onnxruntime.** { *; }
-dontwarn ai.onnxruntime.**
```

## 4. ONNX Runtime의 장점

### 4.1 성능 및 최적화
- Microsoft의 모바일 최적화 지원
- 자동 하드웨어 가속 (NNAPI, XNNPACK)
- 메모리 효율성 개선

### 4.2 개발 편의성  
- Hugging Face 모델 직접 변환 지원
- 복잡한 TFLite 변환 과정 불필요
- 더 나은 디버깅 및 프로파일링 도구

### 4.3 모델 호환성
- PyTorch/Transformers 모델 직접 지원
- 최신 LLM 아키텍처 지원
- ONNX 표준 준수

## 5. 실행 시 고려사항

### 5.1 메모리 관리
ONNX Runtime에서는 명시적으로 리소스를 해제해야 합니다:
```kotlin
inputTensor.close()
attentionTensor.close()
outputs.forEach { it.close() }
```

### 5.2 하드웨어 가속
Android에서 사용 가능한 실행 제공자:
- CPU (기본값)
- NNAPI (Android Neural Networks API)
- XNNPACK (경량화된 추론)

### 5.3 성능 튜닝
```kotlin
val sessionOptions = SessionOptions()
sessionOptions.setIntraOpNumThreads(4)  // CPU 스레드 수
sessionOptions.addConfigEntry("session.inter_op_thread_pool_size", "4")
```

## 6. 문제 해결

### 6.1 일반적인 오류들

**메모리 부족 오류:**
- 모델 양자화 고려
- 배치 크기 줄이기
- 시퀀스 길이 제한

**변환 실패:**
```bash
# 의존성 재설치
pip uninstall torch transformers optimum
pip install torch transformers optimum[onnxruntime]
```

**ONNX 모델 로드 실패:**
- 파일 경로 확인
- 파일 크기 확인 (assets 폴더 제한)
- aaptOptions에 "onnx" 확장자 추가

### 6.2 성능 최적화 팁

1. **모델 양자화:**
```python
# 양자화된 ONNX 모델 생성
from optimum.onnxruntime import ORTQuantizer
quantizer = ORTQuantizer.from_pretrained(ort_model)
quantizer.quantize(save_dir="./quantized_model")
```

2. **실행 제공자 선택:**
```kotlin
sessionOptions.addConfigEntry("providers", "CPUExecutionProvider")
// 또는
sessionOptions.addConfigEntry("providers", "NnapiExecutionProvider,CPUExecutionProvider")
```

3. **배치 추론:**
- 여러 요청을 배치로 처리
- 메모리 사용량과 지연시간 균형

## 7. 대안적 접근법

ONNX Runtime Mobile이 너무 복잡하다면:

### 7.1 더 작은 모델 사용
- TinyLlama-1.1B (더 작음)
- DistilGPT-2 (현재 GPT-2 개선)
- Gemma-2B (Google의 모바일 최적화)

### 7.2 하이브리드 접근
```kotlin
suspend fun generateResponse(query: String): String {
    return if (isComplexQuery(query) && isNetworkAvailable()) {
        callCloudAPI(query)  // 복잡한 쿼리는 클라우드
    } else {
        generateWithLocalModel(query)  // 간단한 쿼리는 온디바이스
    }
}
```

## 8. 참고 자료

- [ONNX Runtime Mobile 공식 문서](https://onnxruntime.ai/docs/get-started/with-mobile.html)
- [Android에서 ONNX Runtime 사용하기](https://onnxruntime.ai/docs/tutorials/mobile/)
- [Hugging Face Optimum 문서](https://huggingface.co/docs/optimum/index)
- [MobiLlama 모델 페이지](https://huggingface.co/MBZUAI/MobiLlama-05B-Chat) 