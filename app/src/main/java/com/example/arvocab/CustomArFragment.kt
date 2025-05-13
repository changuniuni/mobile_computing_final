package com.example.arvocab

import com.google.ar.core.Config
import com.google.ar.core.Session
import com.google.ar.sceneform.ux.ArFragment

class CustomArFragment : ArFragment() {
    // ✔ 정확한 메서드명, 파라미터, 리턴타입을 사용합니다.
    override fun getSessionConfiguration(session: Session): Config {
        // super로부터 Config 객체를 받아와서
        val config = super.getSessionConfiguration(session)
        // LATEST_CAMERA_IMAGE 모드로 설정하고
        config.updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
        // 그대로 반환만 하면, Sceneform이 내부적으로 session.configure(config)까지 처리해 줍니다.
        return config
    }
}