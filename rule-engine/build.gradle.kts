dependencies {
    implementation(project(":core-signal"))
    implementation(project(":core-features"))

    testImplementation(platform(libs.junit.bom))
    testImplementation(libs.junit.jupiter)
    testRuntimeOnly(libs.junit.platform.launcher)
}
