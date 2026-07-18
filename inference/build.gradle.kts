dependencies {
    implementation(project(":core-signal"))
    implementation(project(":core-features"))
    implementation(libs.onnxruntime)
    implementation(libs.jackson.databind)

    testImplementation(platform(libs.junit.bom))
    testImplementation(libs.junit.jupiter)
    testRuntimeOnly(libs.junit.platform.launcher)
}
