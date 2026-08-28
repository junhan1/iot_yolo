package com.example.rknncamera

import android.content.Context
import android.util.Base64
import org.json.JSONObject

/** 测试人脸库保存在应用私有存储，可随时通过页面清空。 */
class FaceFeatureStore(context: Context) {
    data class Match(val name: String, val score: Float)

    private val preferences = context.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)

    fun save(name: String, feature: ByteArray) {
        val records = records().apply {
            put(name, Base64.encodeToString(feature, Base64.NO_WRAP))
        }
        preferences.edit().putString(KEY_RECORDS, records.toString()).apply()
    }

    fun bestMatch(feature: ByteArray, similarity: (ByteArray, ByteArray) -> Float): Match? =
        records().keys().asSequence()
            .mapNotNull { name ->
                decode(records().optString(name))?.let { stored -> Match(name, similarity(feature, stored)) }
            }
            .maxByOrNull(Match::score)

    fun count(): Int = records().length()

    fun clear() {
        preferences.edit().remove(KEY_RECORDS).apply()
    }

    private fun records(): JSONObject = preferences.getString(KEY_RECORDS, null)
        ?.let(::JSONObject)
        ?: JSONObject()

    private fun decode(value: String): ByteArray? = runCatching {
        Base64.decode(value, Base64.NO_WRAP)
    }.getOrNull()

    private companion object {
        const val PREFERENCES = "face_test_store"
        const val KEY_RECORDS = "records"
    }
}