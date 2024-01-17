int main() {
    int a = 0;
    int b = 1 ? 1 : (a = 2);
    return a;
}
