module "kamikaze_asus_laptop_key_pair" {
  source = "terraform-aws-modules/key-pair/aws"

  key_name   = "kamikaze-asus-laptop"
  public_key = var.asus_laptop_key
}
