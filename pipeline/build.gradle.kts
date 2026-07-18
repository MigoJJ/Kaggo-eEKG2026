dependencies {
    implementation(project(":core-signal"))
    implementation(project(":core-features"))
    implementation(project(":rule-engine"))
    implementation(project(":inference"))
    implementation(project(":persistence"))
    implementation(libs.snakeyaml)
    implementation(libs.onnxruntime)

    testImplementation(platform(libs.junit.bom))
    testImplementation(libs.junit.jupiter)
    testRuntimeOnly(libs.junit.platform.launcher)
}
